#!/usr/bin/env python3
"""Compare the search result with non-evolutionary controls and adaptive dispatch."""
import json, random, statistics, time
from pathlib import Path
import numpy as np
from evolve import BASE, BACKENDS, PROGRAM, N, D, full, compile_genome, make_input, run, oracle, equivalent, plan_bytes

class Direct:
    def __init__(self,op): self.op=op
    def apply(self,x):
        k,d,s=self.op
        if k=='add': x[:,d]+=x[:,s]
        elif k=='sub': x[:,d]-=x[:,s]
        elif k=='addi': x[:,d]+=np.uint8(s%256)
        elif k=='muli': x[:,d]*=np.uint8(s%256)
        elif k=='mov': x[:,d]=x[:,s]
        return x

class Adaptive:
    def __init__(self,thresholds,choices): self.thresholds=thresholds; self.choices=choices
    def choose(self,n):
        # Smallest trained bucket >= current batch size; saturate at last.
        for limit,choice in zip(self.thresholds,self.choices):
            if n<=limit: return choice
        return self.choices[-1]

def decode(g): return tuple(tuple((end,b) for end,b in groups) for groups in g)

def main():
    p=Path('results.json'); r=json.loads(p.read_text()); winner=decode(r['best_genome'])
    plans={'unfused_matrix':compile_genome(BASE),'direct_integer':tuple(tuple(Direct(o) for o in ops) for ops in PROGRAM)}
    plans.update({'fused_'+b:compile_genome(full(b)) for b in BACKENDS})
    plans['evolved']=compile_genome(winner)
    # Adaptive policy: select equivalent per-block plans by observed batch size.
    # Train only on independent synthetic microbenchmarks; never on end-to-end holdouts.
    candidates=[decode(c['genome']) for c in r['candidate_records']]
    thresholds=(1,4,16,64,256,1024,4096); adaptive=[]; choices_log=[]
    rng=np.random.default_rng(211); order_rng=random.Random(211)
    for b in range(len(PROGRAM)):
        distinct=list(dict.fromkeys(g[b] for g in candidates)); alternatives=[]
        for d in distinct:
            gg=list(full('dense')); gg[b]=d; alternatives.append(compile_genome(tuple(gg))[b])
        choices=[]; log=[]
        for size in thresholds:
            x=rng.integers(0,256,(size,D),dtype=np.uint8); x[:,N]=1
            timings=[[] for _ in alternatives]; order=list(range(len(alternatives)))
            for rep in range(7):
                order_rng.shuffle(order)
                for i in order:
                    t=time.perf_counter_ns()
                    for _ in range(3):
                        y=x
                        for k in alternatives[i]: y=k.apply(y)
                    timings[i].append((time.perf_counter_ns()-t)/3)
            i=min(range(len(alternatives)),key=lambda j:statistics.median(timings[j]))
            choices.append(alternatives[i]); log.append({'limit':size,'gene':distinct[i],'median_ns':statistics.median(timings[i])})
        adaptive.append(Adaptive(thresholds,choices)); choices_log.append(log)
    plans['adaptive']=tuple(adaptive)
    checks=[]; benchmark=[]; keys=list(plans)
    for n in (1,64,512,4096):
        inp=make_input(n,32000+n); gold=oracle(inp,True)
        for name,plan in plans.items(): assert equivalent(run(plan,inp,trace=True),gold,True),name
        checks.append({'contexts':n,'all_modes_match_oracle':True})
        ts={k:[] for k in keys}; order=keys[:]
        for rep in range(13):
            order_rng.shuffle(order)
            for name in order:
                t=time.perf_counter_ns(); run(plans[name],inp); ts[name].append((time.perf_counter_ns()-t)/1e6)
        record={'contexts':n,'modes':{k:{'median_ms':statistics.median(v),'min_ms':min(v),'max_ms':max(v),'samples_ms':v} for k,v in ts.items()}}
        benchmark.append(record)
        print(n, {k:round(statistics.median(v),4) for k,v in ts.items()},flush=True)
    inp=make_input(256,199); gold=oracle(inp); aborts=[]
    for b in (2,3):
        v=run(plans['adaptive'],inp,abort=(3,b)); assert v[3]['discarded_blocks']>0 and equivalent(v,gold)
        aborts.append({'block':b,'discarded_contexts':v[3]['discarded_blocks'],'pass':True})
    result={'training':'per-block synthetic data, 7 repeats, three calls each','heldout_checks':checks,'benchmark':benchmark,'adaptive_choices':choices_log,'adaptive_rollback':aborts,'static_plan_bytes':{k:plan_bytes(v) for k,v in plans.items() if k not in ('adaptive','direct_integer')}}
    Path('comparison.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
