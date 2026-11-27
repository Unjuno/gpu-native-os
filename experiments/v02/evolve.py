#!/usr/bin/env python3
"""CPU functional prototype of hierarchical matrix ISA; not a GPU timing model."""
import argparse, hashlib, json, os, platform, random, statistics, time, tracemalloc
from pathlib import Path
import numpy as np

N=8; D=N+1; MOD=256; HALT=-1
I=np.eye(D,dtype=np.uint8)
# Affine op format: (kind, destination, source_or_constant, optional ignored).
# Each block terminates at an observable commit boundary.
PROGRAM=(
 (('addi',0,3),('add',1,0),('muli',2,3),('add',2,1),('mov',3,2),('addi',3,11),('sub',4,3),('add',0,4),('addi',1,7)),
 (('add',4,5),('muli',4,5),('add',2,4),('addi',2,17),('mov',6,2),('sub',3,6),('add',1,3),('addi',0,1)),
 (('add',3,2),('addi',3,9),('muli',3,7),('add',4,3),('mov',6,4),('sub',0,6),('add',1,0),('addi',5,13)),
 (('add',4,2),('muli',4,3),('sub',4,5),('addi',4,19),('mov',3,4),('add',6,3),('addi',1,5),('sub',0,6)),
 (('add',6,5),('addi',6,23),('add',0,6),('add',1,4),('muli',2,3),('sub',5,3),('addi',4,1),('addi',7,-1)),
)
XOR=np.bitwise_xor(np.arange(256,dtype=np.uint8)[:,None],np.arange(256,dtype=np.uint8)[None,:])

def opcode_matrix(op):
    kind,d,s=op; m=I.copy()
    if kind=='add': m[d,s]=(int(m[d,s])+1)%MOD
    elif kind=='sub': m[d,s]=(int(m[d,s])-1)%MOD
    elif kind=='addi': m[d,N]=s%MOD
    elif kind=='muli': m[d,d]=s%MOD
    elif kind=='mov': m[d,:]=0; m[d,s]=1
    else: raise ValueError(op)
    return m

def compose(ops):
    m=I.copy()
    for op in ops: m=opcode_matrix(op)@m
    return m

# Gene: tuple of (end-exclusive opcode index, backend) per block.
BACKENDS=('dense','delta','sparse')
BASE=tuple(tuple((i+1,'dense') for i in range(len(ops))) for ops in PROGRAM)

def full(backend): return tuple(((len(ops),backend),) for ops in PROGRAM)

def scalar_op(row,op):
    k,d,s=op
    if k=='add': v=row[d]+row[s]
    elif k=='sub': v=row[d]-row[s]
    elif k=='addi': v=row[d]+s
    elif k=='muli': v=row[d]*s
    elif k=='mov': v=row[s]
    else: raise ValueError(op)
    row[d]=v%MOD

class Kernel:
    def __init__(self,m,backend):
        self.backend=backend
        self.changed=np.flatnonzero(np.any(m!=I,axis=1))
        assert N not in self.changed
        if backend=='dense': self.m=m.copy()
        elif backend=='delta': self.m=m[self.changed].copy()
        elif backend=='sparse':
            self.terms=tuple((int(d),tuple((int(s),int(m[d,s])) for s in range(D) if m[d,s])) for d in self.changed)
        else: raise ValueError(backend)
        # Logical packed representation bytes, not Python allocator footprint.
        self.bytes=(1+D*D if backend=='dense' else 2+len(self.changed)*(D+1) if backend=='delta' else 2+sum(2+2*len(terms) for _,terms in self.terms))
    def apply(self,x):
        if self.backend=='dense': return x@self.m.T
        y=x.copy()
        if self.backend=='delta':
            if len(self.changed): y[:,self.changed]=x@self.m.T
        else:
            for d,terms in self.terms:
                v=np.zeros(len(x),dtype=np.uint8)
                for s,c in terms: v+=x[:,s]*np.uint8(c)
                y[:,d]=v
        return y

def compile_genome(genome):
    plans=[]
    # Every kernel is checked on zero and all basis vectors in Z/256.
    # All three backends are module-linear on the homogeneous D coordinates.
    basis=np.vstack((np.zeros((1,D),dtype=np.uint8),I))
    for ops,groups in zip(PROGRAM,genome,strict=True):
        start=0; block=[]; composed=I.copy()
        for end,backend in groups:
            assert start<end<=len(ops)
            m=compose(ops[start:end]); k=Kernel(m,backend)
            assert np.array_equal(k.apply(basis),basis@m.T)
            composed=m@composed; block.append(k); start=end
        assert start==len(ops) and np.array_equal(composed,compose(ops))
        # Independent scalar oracle tests affine semantics on zero and register basis
        # with homogeneous coordinate 1. Affine maps are determined by these points.
        test=np.zeros((N+1,D),dtype=np.uint8); test[:,N]=1; test[1:,:N]=np.eye(N,dtype=np.uint8)
        expected=[]
        for row in test:
            a=list(map(int,row[:N]))
            for op in ops: scalar_op(a,op)
            expected.append(a+[1])
        x=test
        for k in block: x=k.apply(x)
        assert np.array_equal(x,np.array(expected,dtype=np.uint8))
        plans.append(tuple(block))
    return tuple(plans)

def make_input(n,seed,max_loop=8):
    rng=np.random.default_rng(seed)
    r=rng.integers(0,256,(n,D),dtype=np.uint8); r[:,N]=1
    r[:,7]=rng.integers(1,max_loop+1,n,dtype=np.uint8)
    ram=rng.integers(0,256,(n,256),dtype=np.uint8)
    return r,ram

def run(plan,inp,abort=None,trace=False):
    r,ram=(a.copy() for a in inp); n=len(r); pc=np.zeros(n,dtype=np.int8)
    epoch=0; discarded=0; transitions=0; hashes=[]; issued=0
    while np.any(pc!=HALT):
        if epoch>=100: raise RuntimeError('cycle budget exceeded')
        before=pc.copy(); epoch+=1
        for b,kernels in enumerate(plan):
            ids=np.flatnonzero(before==b)
            if not len(ids): continue
            x=r[ids]
            if hasattr(kernels, "choose"): kernels=kernels.choose(len(ids))
            for k in kernels: x=k.apply(x)
            issued+=len(kernels)
            # Private RAM for each context. Indexed load/store is a matrix-free
            # one-hot selection/scatter. XOR is a matrix-column lookup.
            write=None
            if b==0:
                x[:,5]=ram[ids,x[:,0]]; nxt=1
            elif b==1:
                x[:,2]=XOR[x[:,2],x[:,5]]; nxt=np.where(x[:,2]&1,2,3)
            elif b==2:
                write=(x[:,0],x[:,3]); nxt=4
            elif b==3:
                write=(x[:,1],x[:,4]); nxt=4
            else: nxt=np.where(x[:,7]!=0,0,HALT)
            # Interrupt injection at pre-commit: discard this speculative batch.
            # No register, RAM or PC write has happened yet; retry next epoch.
            if abort==(epoch,b):
                discarded+=len(ids); continue
            r[ids]=x
            if write is not None: ram[ids,write[0]]=write[1]
            pc[ids]=nxt; transitions+=len(ids)
        if trace:
            h=hashlib.sha256(r.tobytes()+ram.tobytes()+pc.tobytes()).hexdigest(); hashes.append(h)
    return r,ram,pc,{'epochs':epoch,'committed_blocks':transitions,'discarded_blocks':discarded,'kernel_dispatches':issued,'trace':hashes}

def oracle(inp,trace=False):
    r,ram=(a.copy() for a in inp); n=len(r); pc=np.zeros(n,dtype=np.int8); hashes=[]; epoch=0
    while np.any(pc!=HALT):
        epoch+=1
        if epoch>100: raise RuntimeError('oracle budget')
        for i in range(n):
            b=int(pc[i])
            if b==HALT: continue
            row=list(map(int,r[i,:N]))
            for op in PROGRAM[b]: scalar_op(row,op)
            if b==0: row[5]=int(ram[i,row[0]]); nxt=1
            elif b==1: row[2]^=row[5]; nxt=2 if row[2]%2 else 3
            elif b==2: ram[i,row[0]]=row[3]; nxt=4
            elif b==3: ram[i,row[1]]=row[4]; nxt=4
            else: nxt=0 if row[7] else HALT
            r[i,:N]=row; pc[i]=nxt
        if trace: hashes.append(hashlib.sha256(r.tobytes()+ram.tobytes()+pc.tobytes()).hexdigest())
    return r,ram,pc,{'trace':hashes}

def equivalent(a,b,trace=False):
    return all(np.array_equal(a[i],b[i]) for i in range(3)) and (not trace or a[3]['trace']==b[3]['trace'])

def mutate(g,rng):
    g=[list(x) for x in g]
    for _ in range(rng.randint(1,3)):
        b=rng.randrange(len(g)); groups=g[b]; j=rng.randrange(len(groups)); end,backend=groups[j]; start=groups[j-1][0] if j else 0
        action=rng.choice(['backend','split','merge'])
        if action=='split' and end-start>1:
            mid=rng.randrange(start+1,end); groups[j:j+1]=[(mid,rng.choice(BACKENDS)),(end,backend)]
        elif action=='merge' and len(groups)>1:
            j=min(j,len(groups)-2); groups[j:j+2]=[(groups[j+1][0],rng.choice(BACKENDS))]
        else: groups[j]=(end,rng.choice([k for k in BACKENDS if k!=backend]))
    return tuple(tuple(x) for x in g)

def bench(plan,inp,reps=5):
    samples=[]
    for _ in range(reps):
        t=time.perf_counter_ns(); run(plan,inp); samples.append((time.perf_counter_ns()-t)/1e6)
    return {'median_ms':statistics.median(samples),'min_ms':min(samples),'max_ms':max(samples),'samples_ms':samples}

def plan_bytes(plan): return sum(k.bytes for block in plan for k in block)
def peak(plan,inp):
    tracemalloc.start(); run(plan,inp); _,p=tracemalloc.get_traced_memory(); tracemalloc.stop(); return p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--generations',type=int,default=8); ap.add_argument('--population',type=int,default=8); ap.add_argument('--seed',type=int,default=20261127); ap.add_argument('--out',default='results.json'); args=ap.parse_args()
    assert args.generations>=1 and args.population>=4
    rng=random.Random(args.seed)
    assert all(int(XOR[x,y])==x^y for x in range(256) for y in range(256))
    base=compile_genome(BASE); train=make_input(512,11); truth=oracle(train,True)
    base_result=run(base,train,trace=True); assert equivalent(truth,base_result,True)
    seen={}; history=[]
    def evaluate(g):
        if g not in seen:
            p=compile_genome(g); r=run(p,train,trace=True); assert equivalent(r,truth,True)
            timing=bench(p,train,3); seen[g]=(timing['median_ms'],p,plan_bytes(p))
        return seen[g][0]
    pop=[BASE]+[full(b) for b in BACKENDS]
    while len(pop)<args.population: pop.append(mutate(rng.choice(pop),rng))
    for gen in range(args.generations):
        for g in pop: evaluate(g)
        ranked=sorted(seen,key=lambda g:(seen[g][0],seen[g][2])); elite=ranked[:4]
        best=elite[0]; rec={'generation':gen+1,'unique_candidates':len(seen),'best_train_ms':seen[best][0],'plan_bytes':seen[best][2]}; history.append(rec); print(json.dumps(rec),flush=True)
        pop=elite[:2]
        while len(pop)<args.population:
            for _ in range(100):
                child=mutate(rng.choice(elite),rng)
                if child not in seen and child not in pop: break
            pop.append(child)
    # Re-time finalists to reduce winner's curse; no holdout used for selection.
    finalists=sorted(seen,key=lambda g:seen[g][0])[:6]; finalist_stats=[]
    for g in finalists: finalist_stats.append((bench(seen[g][1],train,7)['median_ms'],g))
    best=min(finalist_stats,key=lambda a:a[0])[1]; opt=seen[best][1]
    # Independent held-out scalar oracle + exact boundary traces for every epoch.
    tests=[]
    for n,seed in [(1,101),(17,102),(128,103),(512,104),(1024,105),(4096,106)]:
        inp=make_input(n,seed); gold=oracle(inp,True); a=run(base,inp,trace=True); b=run(opt,inp,trace=True)
        assert equivalent(gold,a,True) and equivalent(gold,b,True)
        tests.append({'contexts':n,'seed':seed,'epochs':b[3]['epochs'],'committed_blocks':b[3]['committed_blocks'],'pass':True})
    # Rollback actual pre-commit store at epoch 3. Both branch paths verified separately.
    inp=make_input(256,107); gold=run(base,inp); aborts=[]
    for b in (2,3):
        v=run(opt,inp,abort=(3,b)); assert v[3]['discarded_blocks']>0 and equivalent(v,gold)
        aborts.append({'epoch':3,'block':b,'discarded_contexts':v[3]['discarded_blocks'],'pass':True})
    comparisons=[]
    for n in (1,64,512,4096):
        inp=make_input(n,909+n); run(base,inp); run(opt,inp)
        # Interleaved paired samples, order alternates to limit drift.
        ts=[[],[]]; pp=[base,opt]
        for rep in range(11):
            for j in ([0,1] if rep%2==0 else [1,0]):
                t=time.perf_counter_ns(); run(pp[j],inp); ts[j].append((time.perf_counter_ns()-t)/1e6)
        a,b=map(statistics.median,ts)
        ab=run(base,inp)[3]; bb=run(opt,inp)[3]
        comparisons.append({'contexts':n,'base_ms':a,'optimized_ms':b,'speedup':a/b,'base_range_ms':[min(ts[0]),max(ts[0])],'optimized_range_ms':[min(ts[1]),max(ts[1])],'samples_ms':ts,'base_dispatches':ab['kernel_dispatches'],'optimized_dispatches':bb['kernel_dispatches'],'committed_blocks':bb['committed_blocks'],'base_peak_traced_bytes':peak(base,inp),'optimized_peak_traced_bytes':peak(opt,inp)})
    cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
    report={'environment':{'cpu':cpu,'python':platform.python_version(),'numpy':np.__version__,'visible_cpus':os.cpu_count(),'thread_env':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,'clock':'not fixed; wall clock perf_counter_ns'},'seed':args.seed,'train_contexts':512,'generations':args.generations,'population':args.population,'unique_candidates':len(seen),'xor_exhaustive_cases':65536,'all_candidate_affine_checks_passed':True,'all_candidate_train_traces_passed':True,'history':history,'finalists':[{'retimed_ms':t,'genome':g} for t,g in finalist_stats],'baseline_genome':BASE,'best_genome':best,'baseline_plan_bytes':plan_bytes(base),'optimized_plan_bytes':plan_bytes(opt),'holdout':tests,'rollback':aborts,'benchmark':comparisons,'program':PROGRAM,'candidate_records':[{'genome':g,'train_ms':t,'packed_plan_bytes':size} for g,(t,p,size) in seen.items()]}
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'unique':len(seen),'benchmark':[{k:v for k,v in x.items() if k not in ('samples_ms',)} for x in comparisons],'holdout':tests,'rollback':aborts,'bytes':[plan_bytes(base),plan_bytes(opt)],'best_genome':best},indent=2),flush=True)
if __name__=='__main__': main()
