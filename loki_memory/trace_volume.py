"""One authoritative v13 volume with transactional, epoch-fenced ownership."""

import uuid
from pathlib import Path

from .portable import transaction
from .trace_store import TraceStore
from .trace_atlas import TraceAtlas


class TraceVolume:
    def __init__(self, path, identity, body, epoch, store):
        self.path, self.identity, self.body, self.epoch, self.store = (
            Path(path),
            identity,
            body,
            epoch,
            store,
        )

    @classmethod
    def create(cls, path, body, store):
        if not body:
            raise ValueError("empty body")
        identity = uuid.uuid4().hex
        with transaction(path, create=True) as db:
            db.execute(
                "CREATE TABLE trace(id TEXT PRIMARY KEY, body TEXT, epoch INTEGER, state BLOB)"
            )
            db.execute(
                "INSERT INTO trace VALUES(?,?,0,?)", (identity, body, store.state())
            )
        return cls(path, identity, body, 0, store)

    @classmethod
    def resume(cls, path, identity, body, device="cpu", binding=None):
        with transaction(path) as db:
            row = db.execute(
                "SELECT epoch,state FROM trace WHERE id=? AND body=?", (identity, body)
            ).fetchone()
            if row is None:
                raise ValueError("unassigned host")
            epoch, blob = row
            implementation = TraceAtlas if blob[:8] == TraceAtlas.MAGIC else TraceStore
            store = implementation.load(blob, device=device, binding=binding)
            db.execute("UPDATE trace SET epoch=epoch+1 WHERE id=?", (identity,))
        return cls(path, identity, body, epoch + 1, store)

    def _replace(self, db, body, epoch, blob):
        result = db.execute(
            "UPDATE trace SET body=?,epoch=?,state=? WHERE id=? AND body=? AND epoch=?",
            (body, epoch, blob, self.identity, self.body, self.epoch),
        )
        if result.rowcount != 1:
            raise RuntimeError("stale session")

    def checkpoint(self):
        with transaction(self.path) as db:
            self._replace(db, self.body, self.epoch, self.store.state())

    def transfer(self, destination):
        if not destination:
            raise ValueError("empty destination")
        with transaction(self.path) as db:
            # Cut a copy so failed transfers preserve the running chart.
            copy = type(self.store).load(self.store.state())
            copy.cut()
            self._replace(db, destination, self.epoch + 1, copy.state())
        return self.identity
