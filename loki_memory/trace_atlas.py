"""Optional CPU place atlas with an exact spatial index and bounded order memory."""

import numpy as np
import torch

from .trace_store import TraceStore


class TraceAtlas(TraceStore):
    MAGIC = b"LOKIAT13"
    LIMIT = 64 << 20
    TILE = 4

    def __init__(self, channels, capacity=8192, bodies=1, device="cpu", binding=""):
        if torch.device(device).type != "cpu":
            raise ValueError(
                "TraceAtlas uses a host index; use TraceStore for batched CUDA"
            )
        super().__init__(channels, capacity, bodies, device, binding)
        buckets = max(1, min(self.n, 16384))
        for name, shape, dtype, fill in (
            ("triples", (buckets, 4, 3), torch.int64, -1),
            ("frequency", (buckets, 4), torch.int32, 0),
            ("previous", (self.b, 2), torch.int64, -1),
            (
                "events",
                (self.n, 6),
                torch.int64,
                -1,
            ),  # serial, body, trace, start, end, predecessor
            ("tail", (self.b,), torch.int64, -1),
            ("next_event", (1,), torch.int64, 0),
            ("cursor", (1,), torch.int64, 0),
        ):
            self.layout[name] = (shape, dtype)
            setattr(self, name, torch.full(shape, fill, dtype=dtype))
        self.head = np.full(max(32, self.n // 4), -1, np.int32)
        self.chain = np.full(self.n, -1, np.int32)
        self.bucket = np.full(self.n, -1, np.int32)
        self.sole_frame = None
        self.has_unplaced = False
        # A contiguous sensory projection makes cold-start scans bandwidth-bound.
        self.scale = np.r_[np.full(32, 0.32**-0.5), np.full(16, 0.1296**-0.5)].astype(
            "f4"
        )
        self.sensory = np.zeros((self.n, 48), np.float32)
        self.norm = np.zeros(self.n, np.float32)
        self.examined = 0
        if self.fixed_bytes > self.LIMIT:
            raise ValueError("atlas exceeds fixed budget")

    @property
    def fixed_bytes(self):
        return (
            44
            + len(self.header)
            + sum(
                getattr(self, k).numel() * dt.itemsize
                for k, (_, dt) in self.layout.items()
            )
            + self.head.nbytes
            + self.sensory.nbytes
            + self.norm.nbytes
            + self.scale.nbytes
            + self.chain.nbytes
            + self.bucket.nbytes
        )

    def _hash(self, x, y):
        return ((int(x) * 73856093) ^ (int(y) * 19349663)) % len(self.head)

    def _index(self, i):
        x, y = np.floor(self.features.numpy()[i, 48:50].astype("f4") / self.TILE)
        self.sensory[i] = self.features.numpy()[i, :48] * self.scale
        self.norm[i] = self.sensory[i] @ self.sensory[i]
        if self.chart.numpy()[i] == -2:
            self.has_unplaced = True
            return
        dest = self._hash(x, y)
        old = self.bucket[i]
        if old != dest:
            if old >= 0:
                current = self.head[old]
                if current == i:
                    self.head[old] = self.chain[i]
                else:
                    while self.chain[current] != i:
                        current = self.chain[current]
                    self.chain[current] = self.chain[i]
            self.chain[i] = self.head[dest]
            self.head[dest] = i
            self.bucket[i] = dest
        frame = int(self.chart.numpy()[i])
        if self.sole_frame is None:
            self.sole_frame = frame
        elif self.sole_frame != frame:
            self.sole_frame = -1

    def _rebuild(self):
        self.head.fill(-1)
        self.chain.fill(-1)
        self.bucket.fill(-1)
        self.sole_frame = None
        self.has_unplaced = False
        for i in np.flatnonzero(self.visits.numpy()):
            self._index(i)

    @classmethod
    def load(cls, *args, **kwargs):
        obj = super().load(*args, **kwargs)
        obj._rebuild()
        return obj

    def _indexed(self, key, pos, ctx, body, landmark, k, admit=False, scope="placed"):
        r = int(body[0])
        if (
            not landmark
            and bool(self.located[r])
            and self.sole_frame == int(self.frame[r])
        ):
            p = pos.numpy()[0] + self.offset.numpy()[r]
            x, y = np.floor(p / self.TILE).astype(np.int64)
            seen = set()
            slots = []
            for radius in (1, 2, 4):
                for i in range(x - radius, x + radius + 1):
                    for j in range(y - radius, y + radius + 1):
                        h = self._hash(i, j)
                        if h in seen:
                            continue
                        seen.add(h)
                        t = self.head[h]
                        while t >= 0:
                            slots.append(t)
                            t = self.chain[t]
                result = super()._cpu_query(
                    key, pos, ctx, body, False, k, np.asarray(slots, np.intp)
                )
                self.examined = len(slots)
                if admit:
                    return result  # any omitted point is at least one tile away
                boundary = min(
                    p[0] - (x - radius) * self.TILE,
                    (x + radius + 1) * self.TILE - p[0],
                    p[1] - (y - radius) * self.TILE,
                    (y + radius + 1) * self.TILE - p[1],
                )
                q = min(max(2, k), self.n)
                if k >= q and result[0][0, q - 1] >= 0:
                    sim = float(result[1][0, q - 1])
                    if sim > 0 and -2 * np.log(sim) + 1e-4 < (boundary / 0.65) ** 2:
                        return result
        live = self.visits.numpy() != 0
        if scope == "unplaced":
            live &= self.chart.numpy() == -2
        elif scope == "placed" and self.has_unplaced:
            live &= self.chart.numpy() >= 0
        self.examined = int(np.count_nonzero(live))
        if self.examined:
            probe = np.concatenate((key.numpy()[0], ctx.numpy()[0])) * self.scale
            length = probe @ probe
            approx = self.norm + length - 2 * np.einsum("ij,j->i", self.sensory, probe)
            approx[~live] = np.inf
            q = min(max(2, k), self.examined)
            pilot = np.argpartition(approx, q - 1)[:q]
            upper = approx[pilot]
            if not landmark and bool(self.located[r]):
                delta = self.features.numpy()[pilot, 48:50].astype("f4") - (
                    pos.numpy()[0] + self.offset.numpy()[r]
                )
                upper = upper + np.sum(delta * delta, axis=1) / 0.4225 * (
                    self.chart.numpy()[pilot] == int(self.frame[r])
                )
            kth = upper.max()
            # Round-off bound for two 48-term dot products and rescaling, on both
            # sides of the ranking threshold. Recompute survivors in canonical units.
            error = (
                256
                * np.finfo(np.float32).eps
                * (np.sqrt(self.norm.max()) + np.sqrt(length)) ** 2
            )
            active = np.flatnonzero(approx <= kth + error + 1e-5)
            return super()._cpu_query(key, pos, ctx, body, landmark, k, active)
        return super()._cpu_query(
            key, pos, ctx, body, landmark, k, np.empty(0, np.intp)
        )

    def _cpu_query(self, key, pos, ctx, body, landmark, k):
        return self._indexed(key, pos, ctx, body, landmark, k)

    def _admission(self, key, pos, ctx, rows):
        return self._indexed(key, pos, ctx, rows, False, 1, True)

    def _event(self, body, token, clock):
        events = self.events.numpy()
        tail = self.tail.numpy()
        serial = int(tail[body])
        if token < 0:
            tail[body] = -1
            return
        if serial >= 0:
            prev = events[serial % self.n]
            if prev[0] != serial or clock > prev[4] + 1:
                serial = -1
            elif prev[2] == token:
                prev[4] = clock
                return
        seq = self.next_event.numpy()
        index = int(seq[0])
        events[index % self.n] = (index, body, token, clock, clock, serial)
        tail[body] = index
        seq[0] += 1

    def recall_episode(self, body, clock, steps=8):
        """Oldest first, through the requested beat. Empty for expired or missing evidence."""
        if not 0 <= body < self.b or steps < 1:
            raise ValueError("invalid episode query")
        e = self.events.numpy()
        hits = np.flatnonzero(
            (e[:, 1] == body) & (e[:, 3] <= clock) & (e[:, 4] >= clock)
        )
        if not len(hits):
            return []
        serial = int(e[hits, 0].max())
        out = []
        while serial >= 0 and len(out) < steps:
            stored, who, token, start, end, previous = map(int, e[serial % self.n])
            if stored != serial or who != body:
                break
            idx = token % self.n
            if (
                self.visits.numpy()[idx] == 0
                or self.generation.numpy()[idx] != token // self.n
            ):
                break
            out.append((token, start, min(end, clock)))
            serial = previous
        return out[::-1]

    def _query(self, key, pos, ctx, body, landmark=False, k=8):
        if len(key) > 1 and self.has_unplaced:
            ranked = [
                self._indexed(
                    key[i : i + 1],
                    pos[i : i + 1],
                    ctx[i : i + 1],
                    body[i : i + 1],
                    landmark,
                    k,
                )
                for i in range(len(key))
            ]
            return tuple(torch.cat(group) for group in zip(*ranked))
        return super()._query(key, pos, ctx, body, landmark, k)

    def cue(self, key, pos, ctx, *, body=None, k=8, landmark=False):
        if pos is not None:
            return super().cue(key, pos, ctx, body=body, k=k, landmark=landmark)
        if k < 1 or not len(key):
            raise ValueError("nonempty query and positive k required")
        key, placeholder, ctx, rows = self._inputs(
            key, torch.zeros((len(key), 2)), ctx, body
        )
        ranked = [
            self._indexed(
                key[i : i + 1],
                placeholder[i : i + 1],
                ctx[i : i + 1],
                rows[i : i + 1],
                True,
                k,
                scope="all",
            )
            for i in range(len(key))
        ]
        return tuple(torch.cat(group) for group in zip(*ranked))

    def write(self, key, pos, ctx, rise, drop, clock, live, *, body=None):
        if pos is None:
            key, placeholder, ctx, rows = self._inputs(
                key, torch.zeros((len(key), 2)), ctx, body
            )
            rise = self._float(rise, self.c)
            drop = torch.as_tensor(drop, dtype=torch.float32).reshape(-1).detach()
            live = torch.as_tensor(live, dtype=torch.bool).reshape(-1)
            if (
                not len(key) == len(rise) == len(drop) == len(live)
                or not len(key)
                or not drop.isfinite().all()
                or int(clock) < int(self.meta[0])
            ):
                raise ValueError("invalid sensory write")
            ids = []
            for i, r in enumerate(rows):
                sl = slice(i, i + 1)
                args = key[sl], placeholder[sl], ctx[sl], rows[sl]
                matched = self._indexed(*args, True, 1, scope="unplaced")
                token = self._cpu_write(
                    *args,
                    rise[sl],
                    drop[sl],
                    int(clock),
                    live[sl],
                    matched,
                    unplaced=True,
                ).item()
                self._event(int(r), token, int(clock))
                if token >= 0:
                    self._index(token % self.n)
                ids.append(token)
            return torch.tensor(ids)
        ids = super().write(key, pos, ctx, rise, drop, clock, live, body=body)
        rows = range(len(ids)) if body is None else body
        for r, token in zip(rows, ids.tolist()):
            self._event(int(r), token, int(self.meta[0]))
            if token >= 0:
                self._index(token % self.n)
        return ids

    def observe(self, key, pos, ctx, rise, drop, clock, live, previous, *, body=None):
        """Cue, write and link in one serialized tick, sharing the exact placed ranking."""
        if pos is None or len(key) != 1:
            evidence = self.cue(key, pos, ctx, body=body)
            now = self.write(key, pos, ctx, rise, drop, clock, live, body=body)
        else:
            args = self._inputs(key, pos, ctx, body)
            rise = self._float(rise, self.c)
            drop = torch.as_tensor(drop, dtype=torch.float32).detach().reshape(-1)
            live = torch.as_tensor(live, dtype=torch.bool).reshape(-1)
            if (
                len(rise) != 1
                or len(drop) != 1
                or len(live) != 1
                or not torch.isfinite(drop).all()
                or int(clock) < int(self.meta[0])
            ):
                raise ValueError("invalid atlas tick")
            evidence = self._indexed(*args, False, 8)
            now = self._cpu_write(*args, rise, drop, int(clock), live, evidence)
            token = int(now[0])
            self._event(int(args[-1][0]), token, int(clock))
            if token >= 0:
                self._index(token % self.n)
        self.link(previous, now, live, body=body)
        return evidence, now

    def cut(self, body=None):
        super().cut(body)
        if body is None:
            self.tail.fill_(-1)
            self.previous.fill_(-1)
        else:
            self.tail[body] = -1
            self.previous[body] = -1

    def _motif(self, a, b):
        return ((int(a) * 73856093) ^ (int(b) * 19349663)) % len(self.triples)

    def link(self, id_prev, id_now, live, *, body=None):
        if len(id_now) == 1:
            r = 0 if body is None else int(body[0])
            first, last = int(id_prev[0]), int(id_now[0])
            keep = (
                bool(live[0]) and not bool(self.boundary[r]) and min(first, last) >= 0
            )
            if keep:
                i, j = first % self.n, last % self.n
                g = self.generation.numpy()
                v = self.visits.numpy()
                c = self.chart.numpy()
                keep = (
                    v[i]
                    and v[j]
                    and g[i] == first // self.n
                    and g[j] == last // self.n
                    and c[i] == c[j]
                )
            super().link(id_prev, id_now, live, body=body)
            self._transition(r, first, last, keep)
            return
        rows = self._bodies(len(id_now), body).tolist()
        a, b = (
            torch.as_tensor(x, dtype=torch.long).reshape(-1) for x in (id_prev, id_now)
        )
        i, ok = self._valid(a)
        j, valid = self._valid(b)
        valid &= (
            ok
            & torch.as_tensor(live, dtype=torch.bool)
            & (self.chart[i] == self.chart[j])
        )
        valid &= ~self.boundary[rows]
        super().link(a, b, valid, body=body)
        for r, start, end, keep in zip(rows, a.tolist(), b.tolist(), valid.tolist()):
            self._transition(r, start, end, keep)

    def _transition(self, r, start, end, keep):
        if not keep:
            self.previous[r] = -1
            return
        if start == end:
            return
        first, last = self.previous[r].tolist()
        if first >= 0 and last == start:
            bucket = self._motif(first, start)
            triple = self.triples.numpy()[bucket]
            count = self.frequency.numpy()[bucket]
            sample = (first, start, end)
            hits = np.flatnonzero((triple == sample).all(1))
            idx = int(hits[0]) if len(hits) else int(count.argmin())
            if len(hits):
                count[idx] = min(2**31 - 1, int(count[idx]) + 1)
            else:
                triple[idx] = sample
                count[idx] = 1
        self.previous.numpy()[r] = (start, end)

    def continuations(self, first, second):
        row = self._motif(first, second)
        t = self.triples[row]
        _, valid = self._valid(t)
        keep = (t[:, 0] == first) & (t[:, 1] == second) & valid.all(1)
        order = torch.argsort(self.frequency[row, keep], descending=True)
        return t[keep, 2][order], self.frequency[row, keep][order]

    def role(self, ids):
        """Local statistics that can agree across unrelated appearances and IDs."""
        i, valid = self._valid(torch.as_tensor(ids, dtype=torch.long))
        j, ok = self._valid(self.nbr[i])
        counts = self.count[i].float() * ok
        prob = counts / counts.sum(-1, keepdim=True).clamp_min(1)
        onward = ((self.count[j] > 0).sum(-1) * prob).sum(-1) / 8
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(-1) / np.log(8)
        signature = torch.stack(
            ((counts > 0).sum(-1) / 8, entropy, onward, self.features[i, 50].float()),
            -1,
        )
        return (
            torch.cat(
                (
                    signature,
                    self.relief[i].float(),
                    (prob[..., None] * self.relief[j]).sum(-2),
                ),
                -1,
            )
            * valid[..., None]
        )

    def consolidate(self, budget_ms=2000):
        if self.n <= 4096:
            result = super().consolidate(budget_ms)
            if result["freed"]:
                self._rebuild()
            return result
        # Bounded rolling sweep. Only isolated, weak traces can be pruned here.
        from time import perf_counter

        start = perf_counter()
        begin = int(self.cursor[0])
        end = min(self.n, begin + 512)
        report = dict(
            merged=0, pruned=0, freed=0, value_sweeps=0, committed=False, scanned=0
        )
        if budget_ms >= 1:
            sl = slice(begin, end)
            v = self.visits[sl]
            nbr = self.nbr[sl]
            j, ok = self._valid(nbr)
            p = self.count[sl].float() * ok
            p /= p.sum(1, keepdim=True).clamp_min(1)
            val = torch.maximum(
                self.relief[sl].float(), 0.97 * (p[..., None] * self.value[j]).sum(1)
            )
            # Retain incoming endpoints. Generation checks make removed IDs invalid.
            dead = (
                (v > 0)
                & (v <= 2)
                & (self.relief[sl].amax(1) == 0)
                & (self.features[sl, 50] == 0)
                & ((self.meta[0] - self.last[sl]) >= 32)
                & ~ok.any(1)
            )
            ids = np.flatnonzero(dead.numpy()) + begin
            if len(ids):
                targets = self.nbr.numpy()
                incoming = targets[(targets >= 0)] % self.n
                ids = ids[~np.isin(ids, incoming)]
            # Compare only overlapping x intervals within this maintenance window.
            # Linked vertices stay intact; no path rewiring is needed in a partial sweep.
            points = self.features.numpy()[begin:end].astype("f4")
            occupied = np.flatnonzero((v.numpy() > 0) & ~ok.any(1).numpy())
            ordered = occupied[np.argsort(points[occupied, 48], kind="stable")]
            xs = points[ordered, 48]
            pairs = []
            taken = set()
            for order, i in enumerate(ordered):
                if i in taken:
                    continue
                stop = np.searchsorted(xs, xs[order] + 0.76, side="right")
                cand = ordered[order + 1 : stop]
                if not len(cand):
                    continue
                diff = points[cand] - points[i]
                distance = (
                    np.sum(diff[:, :32] ** 2, 1) / 0.32
                    + np.sum(diff[:, 32:48] ** 2, 1) / 0.1296
                    + np.sum(diff[:, 48:50] ** 2, 1) / 0.4225
                )
                near = cand[
                    (distance < 1.35)
                    & (self.chart.numpy()[begin + cand] == int(self.chart[begin + i]))
                ]
                for j in near:
                    if j not in taken:
                        pairs.append((begin + i, begin + j))
                        taken.update((i, j))
                        break
            if pairs:
                targets = self.nbr.numpy()
                incoming = set((targets[targets >= 0] % self.n).tolist())
                pairs = [
                    (i, j) for i, j in pairs if i not in incoming and j not in incoming
                ]
                ids = np.setdiff1d(ids, np.asarray(pairs, dtype=int).ravel())
            if (perf_counter() - start) * 1000 < budget_ms:
                self.value[sl] = val
                self.cursor[0] = end % self.n
                pruned = len(ids)
                for i, j in pairs:
                    weights = self.visits[[i, j]].float()
                    blend = (self.features[[i, j]].float() * weights[:, None]).sum(
                        0
                    ) / weights.sum()
                    blend[50] = self.features[[i, j], 50].max()
                    self.features[i] = blend
                    self.relief[i] = torch.maximum(self.relief[i], self.relief[j])
                    self.value[i] = torch.maximum(self.value[i], self.value[j])
                    self.last[i] = torch.maximum(self.last[i], self.last[j])
                    self.visits[i] = min(
                        2**31 - 1, int(self.visits[i]) + int(self.visits[j])
                    )
                    self._index(i)
                ids = np.r_[ids, np.asarray([j for _, j in pairs], dtype=int)]
                for i in ids:
                    h = self.bucket[i]
                    if h >= 0:
                        cur = self.head[h]
                        if cur == i:
                            self.head[h] = self.chain[i]
                        else:
                            while self.chain[cur] != i:
                                cur = self.chain[cur]
                            self.chain[cur] = self.chain[i]
                    self.bucket[i] = -1
                    self.chain[i] = -1
                    self.visits[i] = 0
                    self.generation[i] += 1
                    self.value[i] = 0
                    self.nbr[i] = -1
                    self.count[i] = 0
                report.update(
                    merged=len(pairs),
                    pruned=pruned,
                    freed=len(ids),
                    scanned=end - begin,
                    value_sweeps=1,
                    committed=True,
                )
        return dict(report, elapsed_ms=(perf_counter() - start) * 1000)
