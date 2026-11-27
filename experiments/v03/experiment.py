"""Deterministic workload generation, differential tests and bounded search."""
import argparse, hashlib, json, os, platform, random, statistics, time, tracemalloc
from pathlib import Path
import numpy as np
import machine as m


def checks(bits):
    mask=(1<<bits)-1; p=m.workload(bits); rng=random.Random(991+bits)
    cases=0; contexts=0
    def check(p,data,fuel):
        nonlocal cases,contexts
        assert m.decode(m.encode(p,bits))==(tuple(p),bits)
        gold=m.oracle(p,bits,data,fuel)
        groups=m.affine_runs(p)
        for g in ((),tuple((s,e,'dense') for s,e in groups),tuple((s,e,'delta') for s,e in groups)):
            got=m.batch(p,bits,data,fuel,m.compile_plan(p,bits,g))
            if not m.equivalent(gold,got): raise AssertionError((bits,p,fuel,g))
        cases+=1; contexts+=len(data[0])
    data=m.inputs(17,bits,111)
    for fuel in range(91): check(p,data,fuel)
    for q in (
        (('SET',0,mask),('ADDI',0,1),('SUB',0,0),('MULI',1,mask),('ADD',1,1),('HALT',0,0)),
        (('JMP',2,0),('SET',0,0),('ADDI',0,1),('MOV',0,0),('HALT',0,0)),
        (('SET',6,mask),('STORE',6,0),('HALT',0,0)),
        (('SET',6,mask),('LOAD',6,6),('HALT',0,0)),
        (('ADDI',0,1),), (('JMP',0,0),), (('HALT',0,0),),
    ):
        for fuel in (0,1,2,9): check(q,data,fuel)
    for trial in range(80):
        q=[]
        for _ in range(rng.randrange(1,24)):
            op=rng.choice(m.OPS[:-3]); a=rng.randrange(m.REGS)
            b=rng.randrange(mask+1) if op in ('SET','ADDI','MULI','ANDI') else rng.randrange(m.REGS)
            q.append((op,a,b))
        q.append(('HALT',0,0)); check(tuple(q),m.inputs(8,bits,trial),rng.randrange(1,32))
    for bad in (b'',m.encode(p,bits)+b'X',m.encode(p,bits)[:9]+b'\xff'+m.encode(p,bits)[10:]):
        try: m.decode(bad)
        except ValueError: pass
        else: raise AssertionError('bad image accepted')
    dtype=data[0].dtype; basis=np.zeros((m.REGS+1,m.REGS),dtype=dtype);basis[1:]=np.eye(m.REGS,dtype=dtype)
    mem=np.zeros((len(basis),m.RAM),dtype=dtype)
    for s,e in m.affine_runs(p): check(p[s:e]+(('HALT',0,0),),(basis,mem),e-s+1)
    return {'cases':cases,'contexts':contexts,'three_backend_comparisons':cases*3,'invalid_images_rejected':3}


def full(p,mode): return tuple((s,e,mode) for s,e in m.affine_runs(p))
def random_plan(p,rng):
    g=[]
    for s,e in m.affine_runs(p):
        cuts=[s]+[i for i in range(s+1,e) if rng.random()<.45]+[e]
        g.extend((a,b,rng.choice(('dense','delta'))) for a,b in zip(cuts,cuts[1:]))
    return tuple(g)


def mutate(g,rng):
    g=list(g); j=rng.randrange(len(g)); s,e,mode=g[j]; op=rng.randrange(3)
    if op==0 and e-s>1:
        mid=rng.randrange(s+1,e);g[j:j+1]=[(s,mid,mode),(mid,e,rng.choice(('dense','delta'))) ]
    elif op==1 and j+1<len(g) and e==g[j+1][0]: g[j:j+2]=[(s,g[j+1][1],mode)]
    else: g[j]=(s,e,'dense' if mode=='delta' else 'delta')
    return tuple(g)


def measure(fn,n=3):
    samples=[]
    for _ in range(n):
        t=time.perf_counter_ns(); fn(); samples.append(time.perf_counter_ns()-t)
    return samples


def summary(ns):
    return {'median_ms':statistics.median(ns)/1e6,'range_ms':[min(ns)/1e6,max(ns)/1e6],'samples_ns':ns}


def experiment(bits,budget,seed,reps):
    test=checks(bits);p=m.workload(bits); fuel=128;train=m.inputs(128,bits,seed); gold=m.oracle(p,bits,train,fuel)
    records={}; winners={}; history={}
    controls={'direct':(), 'unfused':tuple((i,i+1,'dense') for i,(op,_,_) in enumerate(p) if op in m.AFFINE),'fused_dense':full(p,'dense'),'fused_delta':full(p,'delta')}
    def score(g,cache):
        if g not in cache:
            plan=m.compile_plan(p,bits,g)
            assert m.equivalent(m.batch(p,bits,train,fuel,plan),gold)
            cache[g]=summary(measure(lambda:m.batch(p,bits,train,fuel,plan)))
        return cache[g]['median_ms']
    for name in ('random','evolution'):
        rng=random.Random(seed); cache={}; initial=[controls['fused_dense'],controls['fused_delta']]; trace=[]
        while len(cache)<budget:
            if initial: g=initial.pop(0)
            else:
                for attempt in range(1000):
                    g=random_plan(p,rng) if name=='random' or attempt>=100 else mutate(rng.choice(sorted(cache,key=lambda x:cache[x]['median_ms'])[:4]),rng)
                    if g not in cache: break
                else: raise RuntimeError('search space exhausted')
            score(g,cache);trace.append(min(x['median_ms'] for x in cache.values()))
        finalists=sorted(cache,key=lambda x:cache[x]['median_ms'])[:3]; retimed={}
        for g in finalists:
            plan=m.compile_plan(p,bits,g);retimed[g]=summary(measure(lambda:m.batch(p,bits,train,fuel,plan),5))
        winner=min(retimed,key=lambda x:retimed[x]['median_ms']);winners[name]=winner
        records[name]={'candidates':[{'groups':g,**v} for g,v in cache.items()],'finalists':[{'groups':g,**v} for g,v in retimed.items()],'winner':winner};history[name]=trace
    plans={k:m.compile_plan(p,bits,g) for k,g in (controls|winners).items()}; benches=[]; held=0
    for n in (1,64,512,4096):
        inp=m.inputs(n,bits,seed+1000+n); ref=m.oracle(p,bits,inp,fuel);held+=n
        assert np.all(ref[-1]==m.HALT)
        for name,plan in plans.items():
            if not m.equivalent(m.batch(p,bits,inp,fuel,plan),ref):raise AssertionError(name)
        times={k:[] for k in plans};order=list(plans);rng=random.Random(seed+n)
        for _ in range(reps):
            rng.shuffle(order)
            for k in order:times[k]+=measure(lambda:m.batch(p,bits,inp,fuel,plans[k]),1)
        modes={}
        for k,plan in plans.items():
            tracemalloc.start(); m.batch(p,bits,inp,fuel,plan);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            modes[k]={**summary(times[k]),'peak_traced_bytes':peak,'matrix_data_bytes':sum(t.nbytes+(0 if c is None else c.nbytes) for _,t,c in plan.values())}
        benches.append({'contexts':n,'modes':modes}); print(bits,n,{k:round(v['median_ms'],3) for k,v in modes.items()},flush=True)
    return {'bits':bits,'seed':seed,'differential':test,'heldout_contexts':held,'heldout_modes':len(plans),'search_budget_each':budget,'search':records,'training_history_ms':history,'benchmark':benches}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--budget',type=int,default=24);ap.add_argument('--seed',type=int,default=20261128);ap.add_argument('--reps',type=int,default=9);ap.add_argument('--out',default='results.json');ap.add_argument('--test',action='store_true');a=ap.parse_args()
    if not __debug__ or not 3<=a.budget<=256 or a.reps<3:ap.error('assertions on; budget 3..256; reps >=3')
    if a.test:
        print(json.dumps({b:checks(b) for b in (32,64)},indent=2));return
    out=Path(a.out)
    if out.exists():ap.error('output already exists; choose a new file')
    root=Path(__file__).parent
    env={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,'cpu':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),None) if Path('/proc/cpuinfo').exists() else None,'clock':'unfixed; perf_counter_ns wall time','threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'machine.py',root/'experiment.py')}}
    results={'status':'PASS','environment':env,'reps':a.reps,'widths':[experiment(b,a.budget,a.seed+b,a.reps) for b in (32,64)]}
    out.write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
