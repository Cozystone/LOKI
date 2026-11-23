"""One authoritative local volume and a bounded reverse associative field."""
from contextlib import contextmanager
from pathlib import Path
from functools import wraps
from threading import RLock, get_ident
import hashlib
import sqlite3
import uuid

import numpy as np


def _input(value, shape):
    x = np.asarray(value, dtype=np.float32)
    if x.shape != shape or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError(f"expected normalized {shape} input")
    return x


class PortableField:
    """Fixed visual key, past vision and eight unlabeled normalized sensor components."""
    modes = 32
    size = 64 * (776 + 64) + 51 * 32

    def __init__(self, blob=None):
        if blob is None:
            self.data = np.zeros(self.size, dtype="<f4")
        else:
            if len(blob) != self.size * 4:
                raise ValueError("unexpected field size")
            self.data = np.frombuffer(blob, dtype="<f4").copy()
        self.weights = self.data[:64*776].reshape(64, 776)
        self.precision = self.data[64*776:-51*32].reshape(64, 64)
        self.omega = self.data[-51*32:].reshape(51, 32)
        if blob is None:
            np.fill_diagonal(self.precision, 10)
            self.omega[:] = np.random.default_rng(71).normal(0, 1., self.omega.shape)

    def key(self, frame):
        x = _input(frame, (16, 16, 3))
        tiles = x.reshape(4,4,4,4,3).mean(axis=(1,3))
        mean = tiles.mean(axis=(0,1))
        contrast = tiles - mean
        descriptor = np.r_[contrast.ravel() / (.10 + np.sqrt(np.mean(contrast**2))), (mean-.5)*5]
        phase = descriptor @ self.omega * .55
        return np.r_[np.cos(phase), np.sin(phase)] / np.float32(np.sqrt(self.modes))

    def write(self, frame, past):
        key = self.key(frame)
        v = self.precision @ key
        gain = v / (1 + key @ v)
        self.weights += np.outer(gain, past - .5 - key @ self.weights)
        self.precision -= np.outer(gain, v)

    def recall_signals(self, frame):
        key = self.key(frame)
        signals = (.5 + key @ self.weights[:,768:]).clip(0,1)
        return signals, float(np.clip(1 - .1*(key @ self.precision @ key), 0, 1))

    def recall(self, frame):
        key = self.key(frame)
        result = (.5 + key @ self.weights).clip(0,1)
        support = float(np.clip(1 - .1*(key @ self.precision @ key), 0, 1))
        return result[:768].reshape(16,16,3), result[768:], support


@contextmanager
def transaction(path, *, create=False):
    uri = Path(path).resolve().as_uri() + ("?mode=rwc" if create else "?mode=rw")
    db = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5)
    try:
        db.execute("BEGIN IMMEDIATE")
        yield db
        if db.in_transaction:
            db.commit()
    finally:
        db.close()


def serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            self._idle()
            return method(self, *args, **kwargs)
    return call


class PortableMemory:
    """Use the same physical database on both hosts. A stale writer cannot commit."""
    def __init__(self, path, identity, body, epoch, field):
        self._lock = RLock()
        self._awake = False
        self.path = Path(path).resolve()
        self.identity, self.body, self.epoch = identity, body, epoch
        self._field = field
        self._past = None

    @property
    def state_bytes(self):
        return self._field.data.nbytes + 776*4

    @staticmethod
    def _body(body):
        if not isinstance(body, str) or not body.strip():
            raise ValueError("expected a body instance identifier")
        return body

    @classmethod
    def create(cls, path, body):
        body = cls._body(body)
        identity = uuid.uuid4().hex
        field = PortableField()
        blob = field.data.tobytes()
        with transaction(path, create=True) as db:
            db.execute("CREATE TABLE loki7 (id TEXT PRIMARY KEY, body TEXT, epoch INTEGER, field BLOB, digest BLOB)")
            db.execute("INSERT INTO loki7 VALUES (?,?,0,?,?)", (identity, body, blob, hashlib.sha256(blob).digest()))
        return cls(path, identity, body, 0, field)

    @classmethod
    def resume(cls, path, identity, body):
        body = cls._body(body)
        with transaction(path) as db:
            row = db.execute("SELECT epoch, field, digest FROM loki7 WHERE id=? AND body=?", (identity, body)).fetchone()
            if row is None:
                raise ValueError("wrong identity or assigned host")
            epoch, blob, digest = row
            if hashlib.sha256(blob).digest() != digest:
                raise ValueError("damaged field")
            field = PortableField(blob)
            db.execute("UPDATE loki7 SET epoch=epoch+1 WHERE id=?", (identity,))
        return cls(path, identity, body, epoch+1, field)

    def _check(self, db):
        row = db.execute("SELECT body,epoch FROM loki7 WHERE id=?", (self.identity,)).fetchone()
        if row != (self.body,self.epoch):
            raise RuntimeError("inactive host session")

    @serialized
    def transfer(self, destination):
        destination = self._body(destination)
        with transaction(self.path) as db:
            self._check(db)
            db.execute("UPDATE loki7 SET body=?,epoch=epoch+1 WHERE id=?", (destination,self.identity))
        self._past = None
        return self.identity

    @serialized
    def cut(self):
        self._past = None

    def _idle(self):
        if self._awake:
            raise RuntimeError("use the active wake window")

    @contextmanager
    def wake(self):
        """Stream observations/readouts, committing once; errors roll back the window."""
        with self._lock:
            self._idle()
            with transaction(self.path) as db:
                self._check(db)
                self._awake = True
                stream = _Window(self._field, self._past)
                try:
                    yield stream
                    if stream.dirty:
                        blob = self._field.data.tobytes()
                        db.execute("UPDATE loki7 SET field=?,digest=? WHERE id=?",
                                   (blob,hashlib.sha256(blob).digest(),self.identity))
                    db.commit()
                    self._past = stream.past
                except BaseException:
                    db.rollback()
                    # Discard uncommitted weights even after a partially applied fit.
                    if stream.dirty:
                        blob, = db.execute("SELECT field FROM loki7 WHERE id=?", (self.identity,)).fetchone()
                        self._field = PortableField(blob)
                    raise
                finally:
                    stream.active = False
                    self._awake = False

    def observe(self, frame, signals=None):
        with self.wake() as stream:
            return stream.observe(frame, signals)

    @serialized
    def recall_signals(self, cue):
        with transaction(self.path) as db:
            self._check(db)
            return self._field.recall_signals(cue)

    @serialized
    def recall(self, cue):
        with transaction(self.path) as db:
            self._check(db)
            return self._field.recall(cue)


class _Window:
    def __init__(self, field, past):
        self._field, self.past = field, past
        self.active, self.dirty = True, False
        self.thread = get_ident()

    def _check(self):
        if not self.active or self.thread != get_ident():
            raise RuntimeError("inactive or foreign wake window")

    def observe(self, frame, signals=None):
        self._check()
        x = _input(frame,(16,16,3))
        s = np.zeros(8,np.float32) if signals is None else _input(signals,(8,))
        updated = self.past is not None
        if updated:
            self.dirty = True
            self._field.write(x,self.past)
        self.past = np.r_[x.ravel(),s]
        return updated

    def recall(self, cue):
        self._check()
        return self._field.recall(cue)

    def recall_signals(self, cue):
        self._check()
        return self._field.recall_signals(cue)
