from contextlib import closing
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from loki_memory import PortableMemory
from loki_memory.portable import PortableField


class Migration(unittest.TestCase):
    def test_process_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=(Path(d)/n for n in ("laptop.db","glasses.db"))
            m=PortableMemory.create(a,"A")
            x=np.ones((16,16,3),np.float32)*.2
            m.observe(x); m.observe(1-x)
            ref=m.recall(1-x)[0]
            ident=m.transfer("B")
            a.rename(b)
            script="""
import sys,numpy as np
from loki_memory import PortableMemory
m=PortableMemory.resume(sys.argv[1],sys.argv[2],"B")
x=np.ones((16,16,3),np.float32)*.2
np.save(sys.argv[3],m.recall(1-x)[0])
assert not m.observe(1-x)
assert m.observe(x)
np.save(sys.argv[4],m.recall(x)[0])
m.transfer("A")
"""
            subprocess.run([sys.executable,"-B","-c",script,str(b),ident,str(a)+".npy",str(b)+".npy"],check=True)
            np.testing.assert_array_equal(ref,np.load(str(a)+".npy"))
            b.rename(a)
            with self.assertRaises(RuntimeError):
                m.observe(x)
            n=PortableMemory.resume(a,ident,"A")
            np.testing.assert_array_equal(n.recall(x)[0],np.load(str(b)+".npy"))
            with self.assertRaises(ValueError):
                PortableMemory.resume(a,"other","A")
            with closing(sqlite3.connect(a)) as db, db:
                db.execute("UPDATE loki7 SET digest=zeroblob(32)")
            with self.assertRaises(ValueError):
                PortableMemory.resume(a,ident,"A")

    def test_crash_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"state.db"
            m=PortableMemory.create(path,"A")
            x=np.ones((16,16,3),np.float32)
            m.observe(x); m.observe(1-x)
            prior=m.recall(1-x)[0]
            script="""
import os,sys,numpy as np
from loki_memory import PortableMemory
m=PortableMemory.resume(sys.argv[1],sys.argv[2],"A")
with m.wake() as w:
    x=np.ones((16,16,3),np.float32)
    w.observe(1-x); w.observe(x)
    os._exit(7)
"""
            proc=subprocess.run([sys.executable,"-B","-c",script,str(path),m.identity])
            self.assertEqual(proc.returncode,7)
            resumed=PortableMemory.resume(path,m.identity,"A")
            np.testing.assert_array_equal(prior,resumed.recall(1-x)[0])
            with self.assertRaises(RuntimeError):
                m.recall(x)
            with resumed.wake() as w:
                w.observe(x)
            with self.assertRaises(RuntimeError):
                w.observe(x)

    @unittest.skipUnless(shutil.which("clang") or shutil.which("gcc"),"needs C99 compiler")
    def test_native_learning(self):
        code=r"""
#include <stdio.h>
#include "v7_field.h"
static float field[V7_CELLS], past[776], cue[768], out[777];
int main(void){
 if(fread(field,sizeof(field),1,stdin)!=1) return 1;
 for(int t=0;t<32;t++){
  if(fread(cue,sizeof(cue),1,stdin)!=1 || fread(past,sizeof(past),1,stdin)!=1) return 2;
  loki7_write(field,cue,past);
  out[776]=loki7_recall(field,cue,out);
  if(fwrite(out,sizeof(out),1,stdout)!=1) return 3;
 }
 return 0;
}"""
        # Use binary files rather than text-mode stdio on Windows.
        code=code.replace("int main(void){",'int main(int argc,char**argv){ FILE *in=fopen(argv[1],"rb"), *outf=fopen(argv[2],"wb"); if(!in || !outf)return 4;').replace(",stdin)",",in)").replace(",stdout)",",outf)")
        rng=np.random.default_rng(7)
        f=PortableField()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); exe=root/("probe.exe" if os.name=="nt" else "probe")
            (root/"test.c").write_text(code)
            subprocess.run([shutil.which("clang") or shutil.which("gcc"),"-O2","-std=c99","-I",str(Path(__file__).parent/"native"),str(root/"test.c"),"-lm","-o",str(exe)],check=True)
            expected=[]
            with (root/"inputs").open("wb") as s:
                f.data.tofile(s)
                for _ in range(32):
                    x=rng.random((16,16,3),dtype=np.float32); p=rng.random(776,dtype=np.float32)
                    x.tofile(s); p.tofile(s); f.write(x,p)
                    a,b,v=f.recall(x); expected.append(np.r_[a.ravel(),b,v])
            subprocess.run([str(exe),str(root/"inputs"),str(root/"outputs")],check=True)
            np.testing.assert_allclose(np.fromfile(root/"outputs",np.float32).reshape(32,-1),expected,atol=2e-5)
