"""A bounded stochastic predictive-state filter; simulator state never enters this module."""
import numpy as np


class BeliefModel:
    def __init__(self, revision=0):
        self.emission=np.zeros((96,24),np.float32)
        self.count=np.zeros((4,96,96),np.float32)
        self.n=0
        self.revision=revision
        self.theta=np.zeros((4,53,24),np.float32)
        self.precision=np.repeat((np.eye(53,dtype=np.float32)*8)[None],4,axis=0)
        self.projection=np.random.default_rng(11).normal(0,1.8,(28,24)).astype('f4')
        self.splits=0

    def features(self,x,last):
        v=np.r_[x,last]
        return np.r_[1.,v,np.sin(v@self.projection)].astype('f4')

    def emission_cost(self,x):
        return np.mean((self.emission[:self.n]-x)**2,axis=1)

    def slot(self,x):
        d=self.emission_cost(x)
        if self.n<96 and (not self.n or d.min()>1e-9):
            i=self.n;self.n+=1;self.emission[i]=x
            return i
        return int(d.argmin())

    def fit(self,x,a,y,last=None,topology=True,adapt=True):
        last=np.zeros(4) if last is None else last
        if adapt:
            h=self.features(x,last)
            p=self.precision[a];v=p@h;gain=v/(1+h@v)
            self.theta[a]+=np.outer(gain,y-x-h@self.theta[a])
            p-=np.outer(gain,v)
        if topology:
            i=self.slot(x);j=self.slot(y)
            self.count[a,i,j]+=1
            return i,j

    def probabilities(self):
        c=self.count[:,:self.n,:self.n]
        denom=c.sum(2,keepdims=True)
        return np.where(denom>0,c/np.maximum(denom,1),np.eye(self.n,dtype='f4')[None])

    def infer(self,x,prior=None):
        cost=self.emission_cost(x)
        scale=.000015
        likelihood=np.exp(-(cost-cost.min())/scale)
        posterior=likelihood if prior is None else likelihood*(.998*prior+.002/self.n)
        return posterior/posterior.sum()

    def forecast(self,x,a,last,belief=None):
        local=x+self.features(x,last)@self.theta[a]
        if belief is None or not self.n:return local.clip(0,1)
        trans=self.probabilities()[a]
        prediction=(belief@trans)@self.emission[:self.n]
        evidence=belief@self.count[a,:self.n,:self.n].sum(1)
        w=evidence/(8+evidence)*np.exp(-self.emission_cost(x).min()/.001)
        return np.clip(local*(1-w)+prediction*w,0,1)

    def policy(self,goal):
        t=self.probabilities()
        terminal=self.infer(goal)
        v=terminal.copy()
        for _ in range(32):
            q=.96*(t@v)
            v=np.maximum(terminal,q.max(axis=0))
        return q,t

    def choose(self,b,q,t,active=True):
        utilities=q@b
        if active and np.max(b)<.85:
            # Mutual information between initial hypothesis and observable next view.
            groups=np.unique(self.emission[:self.n].round(4),axis=0,return_inverse=True)[1]
            for a in range(4):
                outcomes=np.zeros((self.n,groups.max()+1))
                for j,g in enumerate(groups):outcomes[:,g]+=t[a,:,j]
                mix=b@outcomes
                info=np.sum(b[:,None]*outcomes*np.log((outcomes+1e-15)/(mix+1e-15)))
                utilities[a]+=.08*info
        return int(np.argmax(utilities))

    def save(self,path):
        with open(path,'wb') as f:
            np.savez(f,emission=self.emission,count=self.count,theta=self.theta,
                     precision=self.precision,projection=self.projection,meta=np.array([self.n,self.revision,self.splits],'<i8'))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            m=cls(int(z['meta'][1]))
            for k in ('emission','count','theta','precision','projection'):
                getattr(m,k)[:]=z[k]
            m.n,m.revision,m.splits=map(int,z['meta'])
            return m
