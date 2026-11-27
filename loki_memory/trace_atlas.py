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
        for i in np.flatnonzero(self.visits.numpy()):
            self._index(i)

    @classmethod
    def load(cls, *args, **kwargs):
        obj = super().load(*args, **kwargs)
        obj._rebuild()
        return obj

    def _indexed(self, key, pos, ctx, body, landmark, k, admit=False):
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
        self.examined = int(np.count_nonzero(self.visits.numpy()))
        if self.examined and (landmark or not bool(self.located[r])):
            probe = np.concatenate((key.numpy()[0], ctx.numpy()[0])) * self.scale
            length = probe @ probe
            approx = self.norm + length - 2 * np.einsum("ij,j->i", self.sensory, probe)
            approx[self.visits.numpy() == 0] = np.inf
            q = min(max(2, k), self.examined)
            kth = np.partition(approx, q - 1)[q - 1]
            # Round-off bound for two 48-term dot products and rescaling, on both
            # sides of the ranking threshold. Recompute survivors in canonical units.
            error = (
                256
                * np.finfo(np.float32).eps
                * (np.sqrt(self.norm.max()) + np.sqrt(length)) ** 2
            )
            active = np.flatnonzero(approx <= kth + error + 1e-5)
            return super()._cpu_query(key, pos, ctx, body, landmark, k, active)
        return super()._cpu_query(key, pos, ctx, body, landmark, k)

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

    def write(self, *args, **kwargs):
        ids = super().write(*args, **kwargs)
        rows = range(len(ids)) if kwargs.get("body") is None else kwargs["body"]
        for r, token in zip(rows, ids.tolist()):
            self._event(int(r), token, int(self.meta[0]))
            if token >= 0:
                self._index(token % self.n)
        return ids

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
            if (perf_counter() - start) * 1000 < budget_ms:
                self.value[sl] = val
                self.cursor[0] = end % self.n
                for i in ids:
                    h = self.bucket[i]
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
                    pruned=len(ids),
                    freed=len(ids),
                    scanned=end - begin,
                    value_sweeps=1,
                    committed=True,
                )
        return dict(report, elapsed_ms=(perf_counter() - start) * 1000)
