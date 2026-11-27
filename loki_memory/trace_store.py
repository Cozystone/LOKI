"""Bounded place/episode table for Going. No action or goal enters recall."""

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from time import perf_counter

import numpy as np
import torch


class TraceStore:
    VERSION = 13
    MAGIC = b"LOKITR13"
    LIMIT = 1 << 20

    def __init__(self, channels, capacity=1024, bodies=64, device="cpu", binding=""):
        self.n, self.b, self.c = map(int, (capacity, bodies, channels))
        if min(self.n, self.b, self.c) < 1:
            raise ValueError("positive dimensions required")
        self.device = torch.device(device)
        self.config = dict(
            channels=self.c,
            capacity=self.n,
            bodies=self.b,
            binding=str(binding),
            version=self.VERSION,
        )
        self.header = json.dumps(
            self.config, sort_keys=True, separators=(",", ":")
        ).encode()
        n, b, c = self.n, self.b, self.c
        self.layout = {
            "features": ((n, 52), torch.float16),  # key, ctx, pos, harm, spare
            "relief": ((n, c), torch.float16),
            "value": ((n, c), torch.float16),
            "visits": ((n,), torch.int32),
            "last": ((n,), torch.int64),
            "chart": ((n,), torch.int64),
            "generation": ((n,), torch.int64),
            "nbr": ((n, 8), torch.int64),
            "count": ((n, 8), torch.int32),
            "frame": ((b,), torch.int64),
            "offset": ((b, 2), torch.float32),
            "located": ((b,), torch.bool),
            "aligned": ((b,), torch.bool),
            "boundary": ((b,), torch.bool),
            "meta": ((3,), torch.int64),
        }  # clock, pressure, last chart number
        size = (
            12
            + len(self.header)
            + 32
            + sum(int(np.prod(sh)) * dt.itemsize for sh, dt in self.layout.values())
        )
        if size > self.LIMIT:
            raise ValueError("contract exceeds 1 MiB")
        for name, (shape, dtype) in self.layout.items():
            setattr(self, name, torch.zeros(shape, dtype=dtype, device=self.device))
        self.nbr.fill_(-1)
        self.frame.copy_(torch.arange(b, device=self.device))
        self.meta[2] = b - 1

    def _float(self, x, width):
        t = torch.as_tensor(x, device=self.device, dtype=torch.float32).detach()
        if t.ndim != 2 or t.shape[1] != width:
            raise ValueError(f"expected batch by {width}")
        if not t.is_cuda and not torch.isfinite(t).all():
            raise ValueError("nonfinite observation")
        return t.contiguous()

    def _bodies(self, length, body):
        ids = (
            torch.arange(length, device=self.device)
            if body is None
            else torch.as_tensor(body, device=self.device, dtype=torch.long).reshape(-1)
        )
        if len(ids) != length or (body is None and length > self.b):
            raise ValueError("body batch mismatch")
        if body is not None and (
            len(set(ids.tolist())) != length or (ids < 0).any() or (ids >= self.b).any()
        ):
            raise ValueError("invalid body indices")
        return ids.contiguous()

    @staticmethod
    def _dist(x, y):
        # mm avoids materializing B x N x D.
        return (
            torch.sum(x * x, 1)[:, None] + torch.sum(y * y, 1)[None, :] - 2 * (x @ y.T)
        ).clamp_min_(0)

    def _query(self, key, pos, ctx, body, landmark=False, k=8):
        if key.is_cuda:
            sensory = self._gpu_cost(key, pos, ctx, body, landmark)
        elif len(key) == 1:
            return self._cpu_query(key, pos, ctx, body, landmark, k)
        else:
            f = self.features.float()
            sensory = self._dist(key, f[:, :32]) / (32 * 0.10**2) + self._dist(
                ctx, f[:, 32:48]
            ) / (16 * 0.09**2)
            absolute = pos + self.offset[body]
            if not landmark:
                same = (self.frame[body, None] == self.chart) & self.located[body, None]
                sensory += self._dist(absolute, f[:, 48:50]) / (0.65**2) * same
            sensory.masked_fill_(self.visits == 0, torch.inf)
        cost, ix = torch.topk(
            sensory, min(max(2, k), self.n), largest=False, sorted=True
        )
        if key.is_cuda:
            from .trace_cuda import launch

            B, W = cost.shape
            K = min(k, W)
            ids = torch.empty((B, K), dtype=torch.long, device=self.device)
            sim = torch.empty((B, K), device=self.device)
            conf = torch.empty(B, device=self.device)
            novel = torch.empty_like(conf)
            launch(
                "finish",
                self.device,
                (B * K + 127) // 128,
                128,
                cost,
                ix,
                self.generation,
                ids,
                sim,
                conf,
                novel,
                self.n,
                K,
                W,
                B,
            )
            return ids, sim, conf, novel
        sim = torch.exp(-0.5 * cost)
        second = sim[:, 1] if self.n > 1 else torch.zeros_like(sim[:, 0])
        conf = sim[:, 0] * (1 - second / (sim[:, 0] + 1e-12)).clamp(0, 1)
        ids = torch.where(torch.isfinite(cost), self.generation[ix] * self.n + ix, -1)
        return ids[:, :k], sim[:, :k], conf, 1 - sim[:, 0]

    def _gpu_check(self, key, pos, ctx, rise=None, drop=None):
        from .trace_cuda import launch

        B = len(key)
        C = 0 if rise is None else self.c
        ok = torch.ones(1, device=self.device, dtype=torch.int32)
        launch(
            "finite_inputs",
            self.device,
            (B * max(32, C) + 127) // 128,
            128,
            key,
            pos,
            ctx,
            key if rise is None else rise,
            pos if drop is None else drop,
            ok,
            B,
            C,
        )
        torch._assert_async(ok, "nonfinite trace observation")

    def _gpu_cost(self, key, pos, ctx, body, landmark=False):
        from .trace_cuda import launch

        B = len(key)
        cost = torch.empty((B, self.n), device=self.device)
        launch(
            "scores",
            self.device,
            (B * self.n + 7) // 8,
            256,
            self.features,
            self.visits,
            self.chart,
            self.frame,
            self.located,
            self.offset,
            body,
            key,
            pos,
            ctx,
            cost,
            self.n,
            B,
            int(landmark),
        )
        return cost

    def _cpu_query(self, key, pos, ctx, body, landmark, k, active=None):
        if active is None:
            active = np.flatnonzero(self.visits.numpy())
        f = self.features.numpy()
        r = int(body[0])
        width = min(self.n, k)
        q = min(max(2, k), len(active))
        ids = np.full(width, -1, np.int64)
        sim = np.zeros(width, np.float32)
        conf = 0.0
        if q:
            ab = pos.numpy()[0] + self.offset.numpy()[r]
            lower = np.sum((f[active, 48:50].astype("f4") - ab) ** 2, axis=1) / 0.4225
            lower *= (
                (self.chart.numpy()[active] == int(self.frame[r]))
                & bool(self.located[r])
                & (not landmark)
            )
            key = key.numpy()[0]
            ctx = ctx.numpy()[0]

            def full(selection):
                idx = active[selection]
                return (
                    lower[selection]
                    + np.sum((f[idx, :32] - key) ** 2, 1) / 0.32
                    + np.sum((f[idx, 32:48] - ctx) ** 2, 1) / 0.1296
                )

            pilot = np.argpartition(lower, q - 1)[:q]
            candidates = np.flatnonzero(lower <= np.max(full(pilot)) + 1e-5)
            exact = full(candidates)
            rank = np.lexsort((active[candidates], exact))[:q]
            slots = active[candidates[rank]]
            v = np.exp(-0.5 * exact[rank])
            used = min(q, width)
            ids[:used] = self.generation.numpy()[slots[:used]] * self.n + slots[:used]
            sim[:used] = v[:used]
            conf = v[0] * (1 - (v[1] if q > 1 else 0) / (v[0] + 1e-12))
        return tuple(
            torch.from_numpy(t)
            for t in (
                ids[None, :],
                sim[None, :],
                np.array([conf], np.float32),
                1 - sim[:1],
            )
        )

    def _inputs(self, key, pos, ctx, body):
        key, pos, ctx = self._float(key, 32), self._float(pos, 2), self._float(ctx, 16)
        if not len(key) == len(pos) == len(ctx):
            raise ValueError("different batch lengths")
        return key, pos, ctx, self._bodies(len(key), body)

    def cue(self, key, pos, ctx, *, body=None, k=8, landmark=False):
        if k < 1:
            raise ValueError("positive k required")
        key, pos, ctx, body = self._inputs(key, pos, ctx, body)
        if key.is_cuda:
            self._gpu_check(key, pos, ctx)
        return self._query(key, pos, ctx, body, landmark, k)

    def _valid(self, ids):
        ix = ids.clamp_min(0) % self.n
        return ix, (ids >= 0) & (self.visits[ix] > 0) & (
            self.generation[ix] == ids // self.n
        )

    def cut(self, body=None):
        rows = self._bodies(self.b if body is None else len(body), body)
        self.frame[rows] = self.meta[2] + torch.arange(
            1, len(rows) + 1, device=self.device
        )
        self.meta[2] += len(rows)
        self.boundary[rows] = True
        self.located[rows] = False
        self.aligned[rows] = False
        self.offset[rows] = 0

    def anchor(self, ids, conf, pos_now, *, body=None):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        ids = ids[:, 0] if ids.ndim == 2 else ids
        pos = self._float(pos_now, 2)
        rows = self._bodies(len(pos), body)
        conf = torch.as_tensor(conf, device=self.device).reshape(-1)
        if ids.shape != (len(pos),) or conf.shape != ids.shape:
            raise ValueError("anchor batch mismatch")
        ix, ok = self._valid(ids)
        # Verify uniqueness without odometry even if caller sent a position-assisted cue.
        f = self.features[ix].float()
        unique, _, margin, _ = self._query(
            f[:, :32].contiguous(), pos, f[:, 32:48].contiguous(), rows, True, 1
        )
        ok &= (
            (self.chart[ix] >= 0)
            & (conf >= 0.7)
            & (margin >= 0.7)
            & (unique[:, 0] == ids)
        )
        target = self.frame[rows]
        self.frame[rows] = torch.where(ok, self.chart[ix], target)
        self.offset[rows] = torch.where(
            ok[:, None], f[:, 48:50] - pos, self.offset[rows]
        )
        self.located[rows] |= ok
        self.aligned[rows] |= ok
        return ok

    def _put(self, name, idx, value):
        field = getattr(self, name)
        field[idx] = torch.as_tensor(value, device=self.device, dtype=field.dtype)

    def write(self, key, pos, ctx, rise, drop, clock, live, *, body=None):
        key, pos, ctx, rows = self._inputs(key, pos, ctx, body)
        rise = self._float(rise, self.c)
        drop = torch.as_tensor(drop, device=self.device, dtype=torch.float32).reshape(
            -1
        )
        live = torch.as_tensor(live, device=self.device, dtype=torch.bool).reshape(-1)
        if not len(rise) == len(key) == len(drop) == len(live):
            raise ValueError("write batch mismatch")
        clock = int(clock)
        if key.is_cuda:
            from .trace_cuda import launch

            self._gpu_check(key, pos, ctx, rise, drop.contiguous())
            torch._assert_async(self.meta[0] <= clock, "invalid clock")
            self.meta[0] = clock
            B = len(key)
            best, ix = self._gpu_cost(key, pos, ctx, rows).min(dim=1)
            out = torch.empty(B, device=self.device, dtype=torch.long)
            pairs = torch.empty((B, B), device=self.device, dtype=torch.uint8)
            launch(
                "pairs",
                self.device,
                (B * B + 127) // 128,
                128,
                self.frame,
                self.offset,
                rows,
                key,
                pos,
                ctx,
                pairs,
                B,
            )
            launch(
                "assign",
                self.device,
                1,
                1,
                best,
                ix,
                self.visits,
                self.generation,
                self.chart,
                self.frame,
                self.located,
                self.offset,
                rows,
                key,
                pos,
                ctx,
                live.contiguous(),
                pairs,
                out,
                self.meta,
                self.n,
                B,
            )
            launch(
                "update",
                self.device,
                B,
                64,
                self.features,
                self.relief,
                self.visits,
                self.last,
                self.chart,
                self.frame,
                self.located,
                self.offset,
                rows,
                key,
                pos,
                ctx,
                rise,
                drop.contiguous(),
                out,
                self.meta,
                self.n,
                B,
                self.c,
            )
            return out
        if clock < 0 or clock < int(self.meta[0]) or not drop.isfinite().all():
            raise ValueError("invalid clock/drop")
        if len(key) == 1:
            return self._cpu_write(key, pos, ctx, rows, rise, drop, clock, live)
        ids, sim, conf, _ = self._query(key, pos, ctx, rows, k=1)
        slot = ids[:, 0].clamp_min(0) % self.n
        match = (sim[:, 0] >= np.exp(-0.5)) & live
        same = self.located[rows] & (self.chart[slot] == self.frame[rows])
        # An unregistered body cannot write its local coordinates into another chart.
        usable = match & same
        result = torch.where(usable, ids[:, 0], -1)
        fresh = (~match) & live
        absolute = pos + self.offset[rows]
        if fresh.any():
            free = (self.visits == 0).nonzero().flatten().tolist()
            batch = torch.cat((key, ctx, absolute), 1).cpu().numpy()
            charts = self.frame[rows].tolist()
            reps = []
            allocated = []
            for i in fresh.nonzero().flatten().tolist():
                selected = -1
                for rep, idx in reps:
                    d = batch[i] - batch[rep]
                    if (
                        charts[rep] == charts[i]
                        and (
                            d[:32] @ d[:32] / 0.32
                            + d[32:48] @ d[32:48] / 0.1296
                            + d[48:] @ d[48:] / 0.4225
                        )
                        <= 1
                    ):
                        selected = idx
                        break
                if selected < 0 and len(reps) < len(free):
                    selected = free[len(reps)]
                    reps.append((i, selected))
                allocated.append(selected)
            dest = torch.as_tensor(allocated, device=self.device)
            result[fresh] = torch.where(
                dest >= 0, self.generation[dest.clamp_min(0)] * self.n + dest, -1
            )
            self.meta[1] += (dest < 0).sum()
        self.meta[0] = clock
        valid = result >= 0
        if valid.any():
            slots = result[valid] % self.n
            unique, inverse, freq = torch.unique(
                slots, return_inverse=True, return_counts=True
            )
            data = torch.cat(
                (
                    key,
                    ctx,
                    absolute,
                    drop[:, None],
                    torch.zeros_like(drop[:, None]),
                    rise,
                ),
                1,
            )[valid]
            acc = torch.zeros((len(unique), data.shape[1]), device=self.device)
            acc.index_add_(0, inverse, data)
            hits = self.visits[unique]
            old = torch.cat((self.features[unique], self.relief[unique]), 1).float()
            prior = hits.clamp_max(31)
            merged = (prior[:, None] * old + acc) / (prior + freq)[:, None]
            self.features[unique] = merged[:, :52].half()
            self.relief[unique] = merged[:, 52:].half()
            self.visits[unique] = (hits.long() + freq).clamp_max(2**31 - 1).int()
            self.last[unique] = clock
            self.chart[slots] = self.frame[rows[valid]]
            self.located[rows[valid]] = True
        return result

    def _admission(self, key, pos, ctx, rows):
        return self._cpu_query(key, pos, ctx, rows, False, 1)

    def _cpu_write(
        self, key, pos, ctx, rows, rise, drop, clock, live, sensory_match=None
    ):
        self.meta.numpy()[0] = clock
        out = np.array([-1], np.int64)
        if not bool(live[0]):
            return torch.from_numpy(out)
        ids, sim, _, _ = (
            self._admission(key, pos, ctx, rows)
            if sensory_match is None
            else sensory_match
        )
        r = int(rows[0])
        frame = self.frame.numpy()[r] if sensory_match is None else -2
        vis = self.visits.numpy()
        if float(sim[0, 0]) >= np.exp(-0.5):
            i = int(ids[0, 0]) % self.n
            if (
                sensory_match is None and not self.located.numpy()[r]
            ) or self.chart.numpy()[i] != frame:
                return torch.from_numpy(out)
        else:
            free = np.flatnonzero(vis == 0)
            if not len(free):
                self.meta.numpy()[1] += 1
                return torch.from_numpy(out)
            i = free[0]
        prior = min(31, int(vis[i]))
        w = 1 / (1 + prior)
        f = self.features.numpy()[i]
        data = np.concatenate(
            (
                key.numpy()[0],
                ctx.numpy()[0],
                pos.numpy()[0] + self.offset.numpy()[r]
                if sensory_match is None
                else np.zeros(2),
                drop.numpy(),
                [0],
            )
        ).astype("f4")
        f[:] = (prior * f.astype("f4") + data) * w
        rel = self.relief.numpy()[i]
        rel[:] = (prior * rel.astype("f4") + rise.numpy()[0]) * w
        vis[i] = min(2147483647, int(vis[i]) + 1)
        self.last.numpy()[i] = clock
        self.chart.numpy()[i] = frame
        if sensory_match is None:
            self.located.numpy()[r] = True
        out[0] = self.generation.numpy()[i] * self.n + i
        return torch.from_numpy(out)

    def link(self, id_prev, id_now, live, *, body=None):
        a, b = [
            torch.as_tensor(x, device=self.device, dtype=torch.long).reshape(-1)
            for x in (id_prev, id_now)
        ]
        mask = torch.as_tensor(live, device=self.device, dtype=torch.bool).reshape(-1)
        if not a.shape == b.shape == mask.shape:
            raise ValueError("link batch mismatch")
        rows = self._bodies(len(a), body)
        if not a.is_cuda and len(a) == 1:
            r = int(rows[0])
            boundary = self.boundary.numpy()
            blocked = boundary[r]
            boundary[r] = False
            first, last = int(a[0]), int(b[0])
            if blocked or not bool(mask[0]) or first < 0 or last < 0 or first == last:
                return
            i, j = first % self.n, last % self.n
            vis = self.visits.numpy()
            gen = self.generation.numpy()
            chart = self.chart.numpy()
            if (
                not vis[i]
                or not vis[j]
                or gen[i] != first // self.n
                or gen[j] != last // self.n
                or chart[i] != chart[j]
            ):
                return
            neighbors = self.nbr.numpy()[i]
            counts = self.count.numpy()[i]
            hist = {int(t): int(v) for t, v in zip(neighbors, counts) if v}
            hist[last] = min(2147483647, hist.get(last, 0) + 1)
            for k, (t, v) in enumerate(
                sorted(hist.items(), key=lambda item: (-item[1], item[0]))[:8]
            ):
                neighbors[k] = t
                counts[k] = v
            return
        mask = mask & ~self.boundary[rows]
        self.boundary[rows] = False
        if a.is_cuda and 0 < len(a) <= 64:
            from .trace_cuda import launch

            launch(
                "edges",
                self.device,
                len(a),
                1,
                a.contiguous(),
                b.contiguous(),
                mask.contiguous(),
                self.visits,
                self.generation,
                self.chart,
                self.nbr,
                self.count,
                self.n,
                len(a),
            )
            return
        i, ok = self._valid(a)
        j, valid = self._valid(b)
        keep = ok & valid & mask & (a != b) & (self.chart[i] == self.chart[j])
        if not keep.any():
            return
        src, inv = torch.unique(i[keep], return_inverse=True)
        neighbors = self.nbr[src].cpu().numpy().copy()
        counts = self.count[src].cpu().numpy().copy()
        groups = inv.cpu().numpy()
        targets = b[keep].cpu().numpy()
        for group in range(len(src)):
            extra = targets[groups == group].tolist()
            hist = {
                int(t): int(v)
                for t, v in zip(neighbors[group], counts[group])
                if t >= 0 and v
            }
            for t in extra:
                hist[t] = hist.get(t, 0) + 1
            pairs = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
            neighbors[group].fill(-1)
            counts[group].fill(0)
            for k, (t, v) in enumerate(pairs):
                neighbors[group, k] = t
                counts[group, k] = min(v, 2**31 - 1)
        self._put("nbr", src, neighbors)
        self._put("count", src, counts)

    def read(self, ids, *, body=None):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        ix, ok = self._valid(ids)
        rows = self._bodies(len(ids), body).reshape((len(ids),) + (1,) * (ids.ndim - 1))
        f = self.features[ix].float()
        rel = ok & (self.chart[ix] == self.frame[rows]) & self.located[rows]
        return dict(
            valid=ok,
            pos_valid=rel,
            pos=torch.where(rel[..., None], f[..., 48:50] - self.offset[rows], 0),
            relief=torch.where(ok[..., None], self.relief[ix], 0),
            harm=torch.where(ok, f[..., 50], 0),
            value=torch.where(ok[..., None], self.value[ix], 0),
            nbr=torch.where(ok[..., None], self.nbr[ix], -1),
            counts=torch.where(ok[..., None], self.count[ix], 0),
        )

    def state(self):
        payload = b"".join(
            getattr(self, k).detach().cpu().contiguous().numpy().tobytes()
            for k in self.layout
        )
        raw = self.MAGIC + struct.pack("<I", len(self.header)) + self.header + payload
        return raw + hashlib.sha256(raw).digest()

    def sha(self):
        return hashlib.sha256(self.state()).hexdigest()

    @classmethod
    def load(cls, data, device="cpu", binding=None):
        if (
            len(data) > cls.LIMIT
            or data[:8] != cls.MAGIC
            or hashlib.sha256(data[:-32]).digest() != data[-32:]
        ):
            raise ValueError("invalid snapshot checksum or format")
        length = struct.unpack_from("<I", data, 8)[0]
        cfg = json.loads(data[12 : 12 + length])
        offset = 12 + length
        if cfg.pop("version") != cls.VERSION or (
            binding is not None and binding != cfg["binding"]
        ):
            raise ValueError("incompatible binding/version")
        obj = cls(**cfg, device=device)
        for name, (shape, dt) in obj.layout.items():
            dest = getattr(obj, name)
            dtype = dest.cpu().numpy().dtype
            arr = np.frombuffer(data, dtype=dtype, count=dest.numel(), offset=offset)
            dest.copy_(torch.from_numpy(arr.copy().reshape(shape)))
            offset += arr.nbytes
        if offset != len(data) - 32:
            raise ValueError("invalid snapshot length")
        return obj

    def save(self, path):
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            tmp = f.name
        try:
            with open(tmp, "wb") as f:
                f.write(self.state())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def consolidate(self, budget_ms=2000):
        """Transactional sleep. A short deadline leaves the live store intact."""
        begin = perf_counter()
        deadline = begin + min(2000.0, max(0.0, float(budget_ms))) / 1000
        report = dict(merged=0, pruned=0, freed=0, value_sweeps=0, committed=False)

        def finish():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return dict(report, elapsed_ms=1000 * (perf_counter() - begin))

        if perf_counter() >= deadline:
            return finish()
        data = {
            k: getattr(self, k).cpu().numpy().copy()
            for k in (
                "features",
                "visits",
                "last",
                "relief",
                "value",
                "chart",
                "generation",
                "nbr",
                "count",
            )
        }
        visits = data["visits"]
        active = np.flatnonzero(visits)
        f = data["features"].astype("f4")
        parent = np.arange(self.n)
        protected = np.maximum(data["relief"].max(1), f[:, 50]) > 0
        order = active[np.lexsort((active, -visits[active], ~protected[active]))]
        merged = pruned = 0
        settled = np.zeros(self.n, bool)
        for i in order:
            if perf_counter() > deadline:
                return finish()
            if parent[i] != i:
                continue
            settled[i] = True
            candidates = active[
                (~settled[active])
                & (parent[active] == active)
                & (active != i)
                & (data["chart"][active] == data["chart"][i])
            ]
            if not len(candidates):
                continue
            # A spatial window first bounds the cost of the high-dimensional comparison.
            dp = np.sum((f[candidates, 48:50] - f[i, 48:50]) ** 2, axis=1) / 0.65**2
            candidates = candidates[dp < 1.35]
            dp = dp[dp < 1.35]
            d = (
                np.mean((f[candidates, :32] - f[i, :32]) ** 2, 1) / 0.10**2
                + np.mean((f[candidates, 32:48] - f[i, 32:48]) ** 2, 1) / 0.09**2
                + dp
            )
            for j in candidates[d < 1.35]:
                ni, nj = int(visits[i]), int(visits[j])
                w = nj / (ni + nj)
                harm = max(f[i, 50], f[j, 50])
                f[i] += (f[j] - f[i]) * w
                f[i, 50] = harm
                data["relief"][i] = np.maximum(data["relief"][i], data["relief"][j])
                data["last"][i] = max(data["last"][i], data["last"][j])
                visits[i] = min(2**31 - 1, ni + nj)
                protected[i] |= protected[j]
                parent[j] = i
                merged += 1
        clock = int(self.meta[0])
        dead = (
            (parent == np.arange(self.n))
            & (visits > 0)
            & (visits <= 2)
            & ~protected
            & (clock - data["last"] >= 32)
        )
        pruned = int(dead.sum())
        nbr = data["nbr"]
        cnt = data["count"]
        gen = data["generation"]
        totals = {}
        for i in active:
            if perf_counter() > deadline:
                return finish()
            src = parent[i]
            if dead[src]:
                continue
            hist = totals.setdefault(src, {})
            for token, weight in zip(nbr[i], cnt[i]):
                if token < 0 or weight <= 0:
                    continue
                k = token % self.n
                if gen[k] != token // self.n or visits[k] == 0:
                    continue
                j = parent[k]
                if j != src and not dead[j]:
                    hist[j] = hist.get(j, 0) + int(weight)
        nbr.fill(-1)
        cnt.fill(0)
        for i, hist in totals.items():
            for k, (j, v) in enumerate(
                sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
            ):
                nbr[i, k] = gen[j] * self.n + j
                cnt[i, k] = min(v, 2**31 - 1)
        released = dead | (parent != np.arange(self.n))
        data["features"][:] = f
        for k in ("features", "visits", "last", "relief", "value", "chart"):
            data[k][released] = 0
        gen[released] += 1
        value = data["relief"].astype("f4")
        reward = value.copy()
        probs = cnt.astype("f4")
        probs /= np.maximum(1, probs.sum(1, keepdims=True))
        ix = nbr.clip(min=0) % self.n
        sweeps = 0
        for _ in range(128):
            if perf_counter() > deadline:
                return finish()
            new = np.maximum(
                reward, 0.97 * np.sum(probs[:, :, None] * value[ix], axis=1)
            )
            sweeps += 1
            if np.max(np.abs(new - value)) < 0.0001:
                break
            value = new
        data["value"][:] = value
        if perf_counter() >= deadline:
            return finish()
        for k, arr in data.items():
            getattr(self, k).copy_(torch.from_numpy(arr))
        report.update(
            merged=merged,
            pruned=pruned,
            freed=merged + pruned,
            value_sweeps=sweeps,
            committed=True,
        )
        return finish()
