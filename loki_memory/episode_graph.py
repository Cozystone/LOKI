"""Optional durable cortical graph. All three tiers commit in one wake transaction."""
import hashlib
import time
import uuid

import numpy as np

from .episodic import EpisodeMemory, _EpisodeWindow, latent
from .portable import transaction, serialized, _input


def pack(x):
    return np.rint(np.clip(x,0,1)*255).astype("u1").tobytes()


def unpack(blob):
    return np.frombuffer(blob,"u1").astype(np.float32)/255


def fingerprint(frame):
    return hashlib.blake2s(pack(_input(frame,(16,16,3))),digest_size=12).digest()


def schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS concept
        (id INTEGER PRIMARY KEY, motor BLOB, effect BLOB, n INTEGER)""")
    db.execute("CREATE INDEX IF NOT EXISTS concept_motor ON concept(motor)")
    db.execute("""CREATE TABLE IF NOT EXISTS episode
        (id INTEGER PRIMARY KEY, prev INTEGER, chain TEXT, host TEXT, stamp INTEGER,
         cue BLOB, pre BLOB, action BLOB, post BLOB, concept INTEGER)""")
    db.execute("CREATE INDEX IF NOT EXISTS episode_chain ON episode(chain,id)")
    db.execute("CREATE INDEX IF NOT EXISTS episode_cue ON episode(host,cue,id)")
    db.execute("CREATE INDEX IF NOT EXISTS episode_prev ON episode(prev)")
    db.execute("CREATE INDEX IF NOT EXISTS episode_concept ON episode(concept,id)")


class _GraphWindow(_EpisodeWindow):
    def _learn(self,key,pending,frame,signals):
        super()._learn(key,pending,frame,signals)
        db=self.db
        pre=latent(pending.anchor[:768].reshape(16,16,3),pending.anchor[768:])
        post=latent(frame,signals)
        effect=post-pre
        motor=np.rint(pending.action*4).astype("u1").tobytes()
        best=None
        for i,blob,n in db.execute("SELECT id,effect,n FROM concept WHERE motor=?",(motor,)):
            mean=np.frombuffer(blob,"<f4")
            distance=np.mean((mean-effect)**2)
            if best is None or distance<best[0]:
                best=(distance,i,mean,n)
        if best is None or best[0]>.015**2:
            cid=db.execute("INSERT INTO concept(motor,effect,n) VALUES(?,?,1)",
                           (motor,effect.astype("<f4").tobytes())).lastrowid
        else:
            _,cid,mean,n=best
            db.execute("UPDATE concept SET n=n+1,effect=? WHERE id=?",
                       ((mean+(effect-mean)/(n+1)).astype("<f4").tobytes(),cid))
        prev=db.execute("SELECT id,post FROM episode WHERE chain=? ORDER BY id DESC LIMIT 1",
                        (self.chain,)).fetchone()
        pre_blob=pack(pre)
        parent=prev[0] if prev and prev[1]==pre_blob and prev[0]+1==int(self._field.clock[0]) else None
        db.execute("INSERT INTO episode VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (int(self._field.clock[0]),parent,self.chain,self.body,time.time_ns(),
                    fingerprint(frame),pre_blob,pack(pending.action),pack(post),cid))


class GraphMemory(EpisodeMemory):
    """Quantized events grow on disk. Read an explicit observation sequence by ordinal."""
    def __init__(self,*args):
        super().__init__(*args)
        self.chain=uuid.uuid4().hex
        with transaction(self.path) as db:
            self._check(db)
            schema(db)

    def _window(self,field,past):
        w=_GraphWindow(field,past)
        w.body,w.chain=self.body,self.chain
        return w

    @serialized
    def cut(self):
        self.chain=uuid.uuid4().hex
        self._past=None

    @serialized
    def encounter(self,frame,host,ordinal):
        if ordinal<1:
            raise ValueError("occurrences start at one")
        with transaction(self.path) as db:
            self._check(db)
            row=db.execute("""SELECT id FROM episode WHERE host=? AND cue=?
                ORDER BY id LIMIT 1 OFFSET ?""",(host,fingerprint(frame),ordinal-1)).fetchone()
            return self._episode(db,row[0]) if row else None

    @staticmethod
    def _episode(db,i):
        row=db.execute("SELECT prev,stamp,pre,action,post,concept,host FROM episode WHERE id=?",(i,)).fetchone()
        if row is None:
            return None
        prev,stamp,pre,act,post,cid,host=row
        following=db.execute("SELECT id FROM episode WHERE prev=?",(i,)).fetchone()
        return dict(id=i,previous=prev,following=following and following[0],time_ns=stamp,
                    host=host,before=unpack(pre),action=unpack(act),after=unpack(post),concept=cid)

    @serialized
    def episode(self,i):
        with transaction(self.path) as db:
            self._check(db)
            return self._episode(db,i)

    @serialized
    def witnessed(self,frame,host,previous,action,after,tolerance=.025):
        """Match a trusted predecessor receipt AND the measured transition, or abstain."""
        action=_input(action,(8,))
        after=_input(after,(20,))
        if previous is None:
            return None
        with transaction(self.path) as db:
            self._check(db)
            rows=db.execute("SELECT id,action,post FROM episode WHERE prev=? AND host=? AND cue=?",
                            (previous,host,fingerprint(frame)))
            matches=[i for i,a,p in rows if np.max(abs(unpack(a)-action))<=tolerance
                     and np.max(abs(unpack(p)-after))<=tolerance]
            return self._episode(db,matches[0]) if len(matches)==1 else None

    @serialized
    def outcomes(self,state,action):
        """Predict a mixture from locally similar antecedents of learned effect types."""
        state=_input(state,(20,))
        action=_input(action,(8,))
        motor=np.rint(action*4).astype("u1").tobytes()
        with transaction(self.path) as db:
            self._check(db)
            options=[]
            for cid,blob in db.execute("SELECT id,effect FROM concept WHERE motor=?",(motor,)):
                refs=db.execute("SELECT pre FROM episode WHERE concept=? ORDER BY id DESC LIMIT 8",(cid,))
                distance=min(np.mean((unpack(pre)-state)**2) for pre, in refs)
                options.append((cid,float(distance),np.clip(state+np.frombuffer(blob,"<f4"),0,1)))
        if not options: return []
        distances=np.array([v[1] for v in options])
        weights=np.exp(-(distances-distances.min())/.0025)
        return [(cid,float(w),pred) for (cid,_,pred),w in zip(options,weights/weights.sum())]
    @serialized
    def reversal_plan(self,last,limit=256):
        """Infer inverse actions from experienced visual returns, then reverse a linked route."""
        with transaction(self.path) as db:
            self._check(db)
            # Aggregate by executed action pair. Both steps must have observable effects.
            pairs=db.execute("""SELECT a.action,b.action,COUNT(*),SUM(a.pre=b.post)
                FROM episode a JOIN episode b ON b.prev=a.id WHERE a.pre<>a.post
                AND b.pre<>b.post AND a.host=b.host AND a.host=?
                GROUP BY a.action,b.action HAVING SUM(a.pre=b.post)>0""",(self.body,))
            inverse={}
            for a,b,n,hits in pairs:
                score=(hits/n,hits)
                if a not in inverse or score>inverse[a][0]:
                    inverse[a]=(score,b)
            plan=[]
            while last is not None:
                row=db.execute("SELECT prev,action,host FROM episode WHERE id=?",(last,)).fetchone()
                if row is None or row[2]!=self.body or row[1] not in inverse or len(plan)>=limit:
                    return None
                last,action,_=row
                plan.append(unpack(inverse[action][1]))
            return plan
