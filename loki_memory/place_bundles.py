"""Fixed-size partial remapping over ATANOR PathOrgan codes (CPU research organ)."""

import torch
from torch import nn


class PlaceBundles(nn.Module):
    def __init__(
        self,
        channels=2,
        bodies=1,
        slots=64,
        bundles=3,
        pooled=False,
        supported_motion=True,
    ):
        super().__init__()
        if min(channels, bodies, slots, bundles) < 1:
            raise ValueError("invalid geometry")
        self.supported_motion = supported_motion
        self.pooled = pooled
        self.register_buffer("reading", torch.tensor(-1, dtype=torch.int64))
        self.register_buffer("splits", torch.tensor(0, dtype=torch.int64))
        self.channels = channels

        # Only C holds spatial prototypes. Contexts hold anonymous outcomes and edge presence.
        def field(name, shape, dtype=torch.float32, fill=0):
            self.register_buffer(name, torch.full(shape, fill, dtype=dtype))

        field("C", (slots, 30))
        field("Pu", (slots, 2))
        field("visits", (slots,))
        field("age", (slots,), torch.int64)
        field("T", (slots, slots))
        field("rates", (bundles, slots, channels + 1))
        field("seen", (bundles, slots))
        field("edges", (bundles, slots, slots), torch.bool)
        field("recent", (bundles,), torch.int64, -1)
        field("fast", (slots, channels + 1))
        field("counts", (slots,))
        field("walk", (slots, slots))
        field("prev", (bodies,), torch.int64, -1)
        field("generation", (slots,), torch.int64)
        field("context_generation", (bundles,), torch.int64)
        field("clock", (), torch.int64)
        field("moves", (), torch.int64)
        field("forgotten", (), torch.int64)

    @property
    def state_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.buffers())

    @staticmethod
    def odometry(code):
        # Invert two projections of the longest PathOrgan scale, relative to birth.
        q = torch.atan2(code[..., 27:29], code[..., 12:14]) * (32 / (2 * torch.pi))
        return torch.stack((q[..., 0], (q[..., 1] - 0.5 * q[..., 0]) / 0.8660254), -1)

    def cut(self):
        self.reading.fill_(-1)
        self.fast.zero_()
        self.counts.zero_()
        self.walk.zero_()
        self.prev.fill_(-1)

    def _recruit(self, p):
        free = torch.where(self.visits == 0)[0]
        if free.numel():
            j = int(free[0])
        else:
            salience = torch.maximum(self.rates.amax((0, 2)), self.fast.amax(1))
            # Bound protection: even the weakest old landmark can eventually be recycled.
            priority = (
                salience
                + (self.visits / (1 + self.visits)) * 0.05
                + 0.15 / (1 + (self.clock - self.age).float() / 32)
            )
            j = int(priority.argmin())
            self.moves.add_(1)
            self.generation[j] += 1
            self.rates[:, j] = 0
            self.seen[:, j] = 0
            self.edges[:, j] = False
            self.edges[:, :, j] = False
            self.fast[j] = 0
            self.counts[j] = 0
            for graph in (self.T, self.walk):
                graph[j] = 0
                graph[:, j] = 0
            self.prev.masked_fill_(self.prev == j, -1)
        self.C[j] = p
        self.Pu[j] = self.odometry(p)
        self.visits[j] = 1
        self.age[j] = self.clock
        return j

    def identify(self):
        active = self.recent >= 0
        if self.pooled and active.any():
            return 0, 1.0
        # Only witnessed positive discrepancies or relief provide positive identity evidence.
        informative = self.fast > 0.15
        if self.supported_motion and informative[:, :-1].any():
            witnessed = self.rates[:, :, -1].amax(0) > 0.15
            informative[:, -1] &= witnessed
        if not informative.any() or not active.any():
            return -1, 0.0
        observed = self.seen > 0
        diff = (self.rates - self.fast).abs()
        cost = torch.where(observed[:, :, None], diff, 1.0)
        costs = (cost * informative).sum((1, 2)) / informative.sum()
        costs.masked_fill_(~active, float("inf"))
        order = costs.argsort()
        best = int(order[0])
        err = float(costs[best])
        rival = float(costs[order[1]]) if order.numel() > 1 else float("inf")
        margin = min(0.4, rival - err)
        support = max(0.0, min(1.0, 1 - err / 0.40, margin / 0.20))
        return (best if support >= 0.7 else -1), support

    @torch.no_grad()
    def observe(self, code, change, motion_error, live=None):
        n = len(self.prev)
        code = torch.as_tensor(code, dtype=self.C.dtype, device=self.C.device)
        change = torch.as_tensor(change, dtype=self.C.dtype, device=self.C.device)
        err = torch.as_tensor(
            motion_error, dtype=self.C.dtype, device=self.C.device
        ).reshape(-1)
        if (
            code.shape != (n, 30)
            or change.shape != (n, self.channels)
            or err.shape != (n,)
        ):
            raise ValueError("incompatible body binding")
        if (
            not torch.isfinite(code).all()
            or not torch.isfinite(change).all()
            or not torch.isfinite(err).all()
        ):
            raise ValueError("nonfinite input")
        mask = (
            torch.ones_like(self.prev, dtype=torch.bool)
            if live is None
            else torch.as_tensor(live, device=code.device, dtype=torch.bool)
        )
        if mask.shape != (n,):
            raise ValueError("live mask mismatch")
        self.prev.masked_fill_(~mask, -1)
        self.clock.add_(1)
        if not mask.any():
            return self.identify()
        p = code[mask]
        out = torch.cat((change[mask].clamp(0, 1), err[mask, None].clamp(0, 1)), 1)
        # Max four insertions per beat. Resolve all assignments again after the last eviction.
        threshold = torch.where(out.amax(1) > 0.15, 0.8, 2.0)
        d = torch.cdist(p, self.C)
        d.masked_fill_(self.visits == 0, float("inf"))
        minimum, u = d.min(1)
        for _ in range(4):
            priority = minimum / threshold
            idx = int(priority.argmax())
            if float(priority[idx]) <= 1:
                break
            j = self._recruit(p[idx])
            d[:, j] = torch.linalg.vector_norm(p - p[idx], dim=1)
            minimum, u = d.min(1)
        accepted = minimum <= threshold
        rows = torch.where(mask)[0]
        self.prev[rows[~accepted]] = -1
        rows, u, out = rows[accepted], u[accepted], out[accepted]
        current = int(self.reading)
        if current >= 0 and not self.pooled:
            contradiction = (
                (out > 0.15)
                & ((self.rates[current, u] - out).abs() > 0.4)
                & (self.seen[current, u, None] > 0)
            )
            if self.supported_motion:
                alternative = self.rates[:, u, -1] > 0.15
                alternative[current] = False
                contradiction[:, -1] &= alternative.any(0)
            if contradiction.any():
                self.sleep()
                self.cut()
                self.splits.add_(1)
        self.fast.scatter_reduce_(0, u[:, None].expand_as(out), out, reduce="amax")
        one = torch.ones_like(u, dtype=self.C.dtype)
        self.counts.index_add_(0, u, one)
        self.visits.index_add_(0, u, one)
        self.age[u] = self.clock
        old = self.prev[rows]
        ok = (old >= 0) & (old != u)
        self.walk.view(-1).index_add_(0, old[ok] * len(self.C) + u[ok], one[ok])
        self.prev[rows] = u
        result = self.identify()
        self.reading.fill_(result[0])
        return result

    @torch.no_grad()
    def sleep(self):
        bundle, support = self.identify()
        if not self.counts.any():
            return -1
        if self.pooled:
            bundle = 0
        if bundle < 0:
            if not (self.fast > 0.15).any():
                return -1
            # An exact tie has no evidence for splitting a new world.
            mask = self.fast > 0.15
            err = (
                torch.where(
                    self.seen[:, :, None] > 0, (self.rates - self.fast).abs(), 1.0
                )
                * mask
            ).sum((1, 2)) / mask.sum()
            compatible = (self.recent >= 0) & (err <= 0.12)
            if compatible.any():
                return -1
            bundle = int(self.recent.argmin())
            if self.recent[bundle] >= 0:
                self.forgotten.add_(1)
                self.context_generation[bundle] += 1
            self.rates[bundle] = 0
            self.seen[bundle] = 0
            self.edges[bundle] = False
        seen = self.counts > 0
        # Fast peaks survive one encounter. Repeated contradicting encounters fade them.
        self.rates[bundle, seen] = torch.maximum(
            0.8 * self.rates[bundle, seen], self.fast[seen]
        )
        self.seen[bundle] += self.counts
        self.T.mul_(0.995).add_(self.walk)
        self.edges[bundle] |= self.walk > 0
        self.recent[bundle] = self.clock
        self.cut()
        return bundle

    def going(self, bundle):
        if not -1 <= bundle < len(self.recent):
            raise ValueError("invalid bundle")
        if bundle < 0 or self.recent[bundle] < 0:
            return (
                self.C,
                torch.zeros_like(self.fast[:, :-1]),
                torch.zeros_like(self.T),
                self.Pu,
            )
        return self.C, self.rates[bundle, :, :-1], self.T * self.edges[bundle], self.Pu

    @torch.no_grad()
    def import_going(self, g):
        """Seed one incumbent map, including a full 64-unit adult. Call only on an empty organ."""
        if self.visits.any():
            raise ValueError("destination already has a map")

        def check(name, shape):
            t = torch.as_tensor(
                getattr(g, name), device=self.C.device, dtype=self.C.dtype
            )
            if t.shape != shape or not torch.isfinite(t).all():
                raise ValueError("incompatible " + name)
            return t

        c = check("C", self.C.shape)
        pu = check("Pu", self.Pu.shape)
        r = check("R", self.fast[:, :-1].shape)
        edges = check("T", self.T.shape)
        cn = check("Cn", self.visits.shape)
        denom = check("Vn", self.visits.shape) if getattr(g, "mended", True) else cn
        used = int(getattr(g, "used", len(c)))
        if (
            not 0 < used <= len(c)
            or (denom < 0).any()
            or (cn < 0).any()
            or (edges < 0).any()
        ):
            raise ValueError("invalid source map")
        self.C[:used] = c[:used]
        self.Pu[:used] = pu[:used]
        self.visits[:used] = cn[:used].clamp_min(1)
        self.T[:used, :used] = edges[:used, :used]
        self.edges[0] = self.T > 0
        self.seen[0, :used] = denom[:used]
        self.rates[0, :used, :-1] = (r[:used] / denom[:used, None].clamp_min(1)).clamp(
            0, 1
        )
        self.recent[0] = 0

    def going_snapshot(self, bundle):
        """Detached copies in Going.refresh's mended R/Vn convention."""
        c, r, t, pu = self.going(bundle)
        n = (
            self.seen[bundle].clone()
            if bundle >= 0 and self.recent[bundle] >= 0
            else torch.zeros_like(self.visits)
        )
        return dict(
            C=c.clone(),
            Pu=pu.clone(),
            T=t.clone(),
            R=r * n[:, None],
            Vn=n,
            Cn=self.visits.clone(),
            Pn=n.clone(),
            used=int((self.visits > 0).sum()),
            generation=self.generation.clone(),
            context_generation=int(self.context_generation[bundle])
            if bundle >= 0
            else -1,
        )
