from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from loki_memory import ReverseMemory


class ReverseTests(unittest.TestCase):
    def setUp(self):
        self.organ=ReverseMemory()
        rng=np.random.default_rng(6012)
        self.old,self.cue=rng.random((2,16,16,3),dtype=np.float32)
        self.levels=[.2,.7]
        self.organ.observe(self.old,self.levels)
        self.organ.observe(self.cue,[.8,.7],1)

    def test_raw_readout_and_bound(self):
        m=self.organ
        self.assertLessEqual(m.state_bytes,262144)
        for name,_ in m.layout:
            self.assertTrue(np.shares_memory(m.matrix,getattr(m,name)))
        self.assertTrue(np.shares_memory(m.core.matrix,m.matrix))
        m.cut()
        before=m.matrix.copy()
        # For one observation and a unit Fourier key, ridge shrinks by 1/(1+lambda).
        np.testing.assert_allclose(m.recall(self.cue),.5+(self.old-.5)/1.1,atol=2e-6)
        np.testing.assert_array_equal(m.pull(self.cue,[1,1],(1,2),base=[1.,2.]),[1.,2.])
        self.assertEqual(m.return_scores(self.cue,(0,1)).shape,(2,))
        np.testing.assert_array_equal(before,m.matrix)

    def test_blackout_cut_and_snapshot(self):
        m=self.organ
        visual=m.visual.copy()
        m.observe(None,self.levels,2)
        m.observe(self.cue,self.levels,3)
        np.testing.assert_array_equal(visual,m.visual)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"memory.npz"
            m.save(path,binding="maze")
            with self.assertRaises(ValueError):
                m.load(path,binding="other")
            restored=m.load(path,binding="maze")
        for organ in (m,restored):
            organ.observe(self.old,self.levels,0)
        np.testing.assert_array_equal(restored.matrix,m.matrix)
        before=m.matrix.copy()
        with self.assertRaises(ValueError):
            m.observe(self.old,[np.nan,1],0)
        np.testing.assert_array_equal(m.matrix,before)

    @unittest.skipUnless(shutil.which("clang") or shutil.which("gcc"),"C99 compiler unavailable")
    def test_portable_readout(self):
        import os
        src = '''
#include <stdio.h>
#include "v6_recall.h"
static float omega[48*12], weights[24*768], cue[768], result[768];
int main(int argc, char **argv) {
    if (argc != 3) return 1;
    FILE *input = fopen(argv[1],"rb");
    if (!input || fread(omega,sizeof(omega),1,input)!=1 ||
        fread(weights,sizeof(weights),1,input)!=1 || fread(cue,sizeof(cue),1,input)!=1) return 2;
    fclose(input);
    loki_v6_recall(omega,weights,cue,result);
    FILE *out = fopen(argv[2],"wb");
    if (!out || fwrite(result,sizeof(result),1,out)!=1) return 3;
    return fclose(out);
}
'''
        include=Path(__file__).resolve().parent/"native"
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"run.c").write_text(src,encoding="ascii")
            exe=p/("run.exe" if os.name=="nt" else "run")
            subprocess.run([shutil.which("clang") or shutil.which("gcc"),"-std=c99","-O2",
                            "-I",str(include),str(p/"run.c"),"-o",str(exe),"-lm"],check=True)
            with (p/"in").open("wb") as f:
                for a in (self.organ.omega[:48],self.organ.visual,self.cue):
                    a.tofile(f)
            subprocess.run([str(exe),str(p/"in"),str(p/"out")],check=True)
            c=np.fromfile(p/"out",dtype=np.float32).reshape(16,16,3)
        np.testing.assert_allclose(c,self.organ.recall(self.cue),atol=2e-6)
