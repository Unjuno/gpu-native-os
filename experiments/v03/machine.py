"""V03: exact integer guest ISA, scalar oracle and batch/matrix CPU model."""
import struct
import numpy as np

REGS, RAM = 8, 16
OPS = ('MOV','SET','ADD','SUB','ADDI','MULI','XOR','ANDI','LOAD','STORE','JNZ','JMP','HALT')
AFFINE = frozenset(OPS[:6])
RUN, HALT, FAULT, BUDGET = 0, 1, 2, 3
HEADER, WORD = struct.Struct('<4sBI'), struct.Struct('<BQQ')


def validate(program, bits):
    if bits not in (32, 64) or not 0 < len(program) < 2**31:
        raise ValueError('invalid width or program length')
    limit = 1 << bits
    for op, a, b in program:
        if op not in OPS or type(a) is not int or type(b) is not int:
            raise ValueError('invalid instruction')
        if op == 'HALT':
            valid = a == b == 0
        elif op == 'JMP':
            valid = 0 <= a < len(program) and b == 0
        elif op == 'JNZ':
            valid = 0 <= a < REGS and 0 <= b < len(program)
        elif op in ('SET','ADDI','MULI','ANDI'):
            valid = 0 <= a < REGS and 0 <= b < limit
        else:
            valid = 0 <= a < REGS and 0 <= b < REGS
        if not valid:
            raise ValueError('invalid operands')
    return tuple(program)


def encode(program, bits):
    p = validate(program, bits)
    return HEADER.pack(b'GNO3', bits, len(p)) + b''.join(WORD.pack(OPS.index(op), a, b) for op, a, b in p)


def decode(blob):
    if len(blob) < HEADER.size:
        raise ValueError('short header')
    magic, bits, n = HEADER.unpack_from(blob)
    if magic != b'GNO3' or len(blob) != HEADER.size + n * WORD.size:
        raise ValueError('invalid image')
    p = []
    for opcode, a, b in WORD.iter_unpack(blob[HEADER.size:]):
        if opcode >= len(OPS):
            raise ValueError('unknown opcode')
        p.append((OPS[opcode], a, b))
    return validate(p, bits), bits


def inputs(n, bits, seed):
    rng = np.random.default_rng(seed)
    dtype = np.dtype('uint'+str(bits))
    r = rng.integers(0, np.iinfo(dtype).max, (n, REGS), dtype=dtype, endpoint=True)
    m = rng.integers(0, np.iinfo(dtype).max, (n, RAM), dtype=dtype, endpoint=True)
    r[:, 6] %= RAM
    r[:, 7] = rng.integers(1, 6, n, dtype=dtype)
    return r, m


def prepare(program, bits, data, fuel):
    p = validate(program, bits)
    r, m = (np.asarray(x) for x in data)
    dtype = np.dtype('uint'+str(bits))
    if r.ndim != 2 or r.shape[1] != REGS or m.shape != (len(r), RAM) or r.dtype != dtype or m.dtype != dtype:
        raise ValueError('invalid guest state')
    if type(fuel) is not int or not 0 <= fuel < 2**63:
        raise ValueError('invalid fuel')
    return p, r.copy(), m.copy(), np.zeros(len(r), np.int64), np.full(len(r), fuel, np.int64), np.zeros(len(r), np.uint8)


def oracle(program, bits, data, fuel):
    p, r, m, pc, left, status = prepare(program, bits, data, fuel)
    mask = (1 << bits)-1
    for i in range(len(r)):
        x, ram = list(map(int,r[i])), list(map(int,m[i]))
        q, f, s = 0, fuel, RUN
        while s == RUN:
            if f == 0: s = BUDGET; break
            if not 0 <= q < len(p): s = FAULT; break
            op,a,b = p[q]; nxt = q+1
            if op == 'MOV': x[a] = x[b]
            elif op == 'SET': x[a] = b
            elif op == 'ADD': x[a] = (x[a]+x[b]) & mask
            elif op == 'SUB': x[a] = (x[a]-x[b]) & mask
            elif op == 'ADDI': x[a] = (x[a]+b) & mask
            elif op == 'MULI': x[a] = (x[a]*b) & mask
            elif op == 'XOR': x[a] ^= x[b]
            elif op == 'ANDI': x[a] &= b
            elif op == 'LOAD':
                if x[b] >= RAM: s = FAULT; break
                x[a] = ram[x[b]]
            elif op == 'STORE':
                if x[a] >= RAM: s = FAULT; break
                ram[x[a]] = x[b]
            elif op == 'JNZ': nxt = b if x[a] else nxt
            elif op == 'JMP': nxt = a
            elif op == 'HALT': s = HALT; nxt = q
            q, f = nxt, f-1
        r[i], m[i], pc[i], left[i], status[i] = x, ram, q, f, s
    return r,m,pc,left,status


def matrix(ops, bits):
    """Column-vector homogeneous affine map over Z/(2**bits)."""
    dtype = np.dtype('uint'+str(bits)); mask = (1<<bits)-1
    eye = np.eye(REGS+1, dtype=dtype); out = eye.copy()
    for op,a,b in ops:
        t = eye.copy()
        if op == 'MOV': t[a] = 0; t[a,b] = 1
        elif op == 'SET': t[a] = 0; t[a,REGS] = b
        elif op == 'ADD': t[a,b] = (int(t[a,b])+1)&mask
        elif op == 'SUB': t[a,b] = (int(t[a,b])-1)&mask
        elif op == 'ADDI': t[a,REGS] = b
        elif op == 'MULI': t[a,a] = b
        else: raise ValueError('non-affine opcode')
        out = t @ out
    return out


def compile_plan(program, bits, groups):
    p = validate(program, bits); plan = {}; end = 0
    eye = np.eye(REGS+1,dtype='uint'+str(bits))
    for start,stop,mode in sorted(groups):
        if not end <= start < stop <= len(p) or mode not in ('dense','delta'):
            raise ValueError('invalid or overlapping group')
        t = matrix(p[start:stop],bits)
        changed = np.flatnonzero(np.any(t != eye,axis=1))
        if mode == 'dense': plan[start] = (stop,t,None)
        else: plan[start] = (stop,t[changed],changed)
        end = stop
    return plan


def batch(program,bits,data,fuel,plan=None):
    p,r,m,pc,left,status = prepare(program,bits,data,fuel)
    plan = {} if plan is None else plan
    while np.any(status == RUN):
        status[(status==RUN)&(left==0)] = BUDGET
        status[(status==RUN)&((pc<0)|(pc>=len(p)))] = FAULT
        active = np.flatnonzero(status==RUN); old = pc[active].copy()
        for q in np.unique(old):
            ids = active[old==q]; q=int(q)
            if q in plan:
                stop,t,changed = plan[q]; count=stop-q
                eligible=left[ids]>=count; good=ids[eligible]
                if len(good):
                    x=np.ones((len(good),REGS+1),dtype=r.dtype); x[:,:REGS]=r[good]
                    y=x@t.T
                    if changed is None: r[good]=y[:,:REGS]
                    else: r[np.ix_(good,changed)]=y
                    pc[good]=stop; left[good]-=count
                ids=ids[~eligible]
            if not len(ids): continue
            op,a,b=p[q]; nxt=q+1
            if op=='MOV': r[ids,a]=r[ids,b]
            elif op=='SET': r[ids,a]=b
            elif op=='ADD': r[ids,a]+=r[ids,b]
            elif op=='SUB': r[ids,a]-=r[ids,b]
            elif op=='ADDI': r[ids,a]+=r.dtype.type(b)
            elif op=='MULI': r[ids,a]*=r.dtype.type(b)
            elif op=='XOR': r[ids,a]^=r[ids,b]
            elif op=='ANDI': r[ids,a]&=r.dtype.type(b)
            elif op in ('LOAD','STORE'):
                addr=r[ids,b if op=='LOAD' else a]; valid=addr<RAM
                status[ids[~valid]]=FAULT; ids=ids[valid]; addr=addr[valid].astype(np.intp)
                if op=='LOAD': r[ids,a]=m[ids,addr]
                else: m[ids,addr]=r[ids,b]
            elif op=='JNZ': nxt=np.where(r[ids,a]!=0,b,nxt)
            elif op=='JMP': nxt=a
            elif op=='HALT': status[ids]=HALT; nxt=q
            pc[ids]=nxt; left[ids]-=1
    return r,m,pc,left,status


def equivalent(a,b):
    return len(a)==len(b) and all(np.array_equal(x,y) for x,y in zip(a,b))


def affine_runs(p):
    runs=[]; start=None
    for i,(op,_,_) in enumerate(p):
        if op in AFFINE and start is None: start=i
        if op not in AFFINE and start is not None: runs.append((start,i)); start=None
    if start is not None: runs.append((start,len(p)))
    return runs


def workload(bits):
    return (('ADDI',0,3),('ADD',1,0),('MULI',2,5),('ADD',2,1),('MOV',3,2),('ADDI',3,17),('SUB',4,3),('ADD',0,4),('LOAD',5,6),('XOR',2,5),('ANDI',5,1),('JNZ',5,15),('ADDI',3,13),('STORE',6,3),('JMP',17,0),('ADDI',4,19),('STORE',6,4),('ADDI',7,(1<<bits)-1),('JNZ',7,0),('HALT',0,0))
