#!/usr/bin/env python3
"""Generate C kernels from the candidate affine matrices, train and benchmark.
The target is the host CPU, not CUDA; C compiler optimization is a strong control.
"""
import ctypes, json, os, random, statistics, subprocess, time
from pathlib import Path
import numpy as np
from evolve import N,D,PROGRAM,BASE,BACKENDS,full,compose,compile_genome,make_input,run,oracle,equivalent
from compare import Adaptive,Direct,decode

DIR=Path(__file__).resolve().parent

def generate(name,b,cuts=None):
    lines=[f'void {name}(uint8_t * restrict x, size_t n) {{',f'  for(size_t i=0;i<n;i++,x+={D}) {{']
    lines += [f'    uint8_t v{j}=x[{j}];' for j in range(D)]
    if cuts is None:
        for k,d,s in PROGRAM[b]:
            expr={'add':f'v{d}+v{s}','sub':f'v{d}-v{s}','addi':f'v{d}+({s%256})','muli':f'v{d}*({s%256})','mov':f'v{s}'}[k]
            lines.append(f'    v{d}=(uint8_t)({expr});')
    else:
        start=0
        for g,end in enumerate(cuts):
            m=compose(PROGRAM[b][start:end]); start=end
            changed=[j for j in range(N) if not np.array_equal(m[j],np.eye(D,dtype=np.uint8)[j])]
            for j in changed:
                terms=[f'{int(m[j,k])}u*v{k}' for k in range(D) if m[j,k]]
                lines.append(f'    uint8_t t{g}_{j}=(uint8_t)({"+".join(terms) or "0"});')
            for j in changed: lines.append(f'    v{j}=t{g}_{j};')
    lines += [f'    x[{j}]=v{j};' for j in range(N)]
    lines += ['  }','}']
    return '\n'.join(lines)

class CKernel:
    def __init__(self,lib,name):
        self.f=getattr(lib,name); self.f.argtypes=[ctypes.c_void_p,ctypes.c_size_t]; self.f.restype=None
    def apply(self,x):
        assert x.dtype==np.uint8 and x.flags.c_contiguous
        self.f(x.ctypes.data,len(x)); return x

def main():
    r=json.loads((DIR/'results.json').read_text()); rng=random.Random(500); nr=np.random.default_rng(500)
    lines=['#include <stdint.h>','#include <stddef.h>']; partitions=[]
    for b in range(len(PROGRAM)):
        cuts=sorted({tuple(x[0] for x in c['genome'][b]) for c in r['candidate_records']}); partitions.append(cuts)
        lines.append(generate(f'direct_{b}',b))
        for j,c in enumerate(cuts): lines.append(generate(f'mat_{b}_{j}',b,c))
    (DIR/'generated.c').write_text('\n\n'.join(lines)+'\n')
    flags=['-O3','-march=native','-fPIC','-shared','-std=c11']; cmd=['gcc',*flags,str(DIR/'generated.c'),'-o',str(DIR/'generated.so')]
    subprocess.run(cmd,check=True); lib=ctypes.CDLL(str(DIR/'generated.so'))
    direct=[]; fused=[]; adaptive=[]; log=[]; thresholds=(1,4,16,64,256,1024,4096)
    for b,cuts in enumerate(partitions):
        direct.append((CKernel(lib,f'direct_{b}'),))
        alternatives=[(CKernel(lib,f'mat_{b}_{j}'),) for j in range(len(cuts))]
        fused.append(alternatives[cuts.index((len(PROGRAM[b]),))])
        # Check basis plus random exact states against the independent dense matrix.
        expected_matrix=compose(PROGRAM[b]); basis=np.vstack((np.zeros((1,D),dtype=np.uint8),np.eye(D,dtype=np.uint8),nr.integers(0,256,(1000,D),dtype=np.uint8)))
        for (k,) in alternatives: assert np.array_equal(k.apply(basis.copy()),basis@expected_matrix.T)
        valid=basis.copy(); valid[:,N]=1
        assert np.array_equal(direct[-1][0].apply(valid.copy()),valid@expected_matrix.T)
        picks=[]; log_b=[]
        for size in thresholds:
            src=nr.integers(0,256,(size,D),dtype=np.uint8); src[:,N]=1
            measurements=[[] for _ in cuts]; order=list(range(len(cuts)))
            for rep in range(11):
                rng.shuffle(order)
                for j in order:
                    (k,)=alternatives[j]; x=src.copy(); t=time.perf_counter_ns()
                    for _ in range(5): k.apply(x)
                    measurements[j].append((time.perf_counter_ns()-t)/5)
            j=min(range(len(cuts)),key=lambda j:statistics.median(measurements[j])); picks.append(alternatives[j]); log_b.append({'size':size,'partition':cuts[j],'median_ns':statistics.median(measurements[j])})
        adaptive.append(Adaptive(thresholds,picks)); log.append(log_b)
    plans={'matrix_reference':compile_genome(BASE),'numpy_direct':tuple(tuple(Direct(o) for o in ops) for ops in PROGRAM),'numpy_fused_delta':compile_genome(full('delta')),'c_direct':tuple(direct),'c_fused_matrix':tuple(fused),'c_adaptive_matrix':tuple(adaptive)}
    benchmarks=[]
    for n in (1,64,512,4096,16384):
        inp=make_input(n,60000+n); gold=oracle(inp,True)
        for name,p in plans.items(): assert equivalent(run(p,inp,trace=True),gold,True),name
        ts={k:[] for k in plans}; order=list(plans)
        for rep in range(13):
            rng.shuffle(order)
            for name in order:
                t=time.perf_counter_ns(); run(plans[name],inp); ts[name].append((time.perf_counter_ns()-t)/1e6)
        row={'contexts':n,'all_match_oracle':True,'modes':{k:{'median_ms':statistics.median(v),'min_ms':min(v),'max_ms':max(v),'samples_ms':v} for k,v in ts.items()}}; benchmarks.append(row)
        print(n,{k:round(statistics.median(v),3) for k,v in ts.items()},flush=True)
    inp=make_input(256,60001); gold=oracle(inp); aborts=[]
    for b in (2,3):
        v=run(plans['c_adaptive_matrix'],inp,abort=(3,b)); assert v[3]['discarded_blocks']>0 and equivalent(v,gold)
        aborts.append({'block':b,'discarded':v[3]['discarded_blocks'],'pass':True})
    result={'gcc':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],'flags':flags,'affinity':sorted(os.sched_getaffinity(0)),'unique_matrix_partitions':sum(map(len,partitions)),'all_compiled_basis_and_random_checks_passed':True,'adaptive_choices':log,'benchmark':benchmarks,'rollback':aborts}
    (DIR/'native.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
