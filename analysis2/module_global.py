# Using a trivial class for module global variables
import multiprocessing as mp
import marshal
import pickle
import types

class TrivialClass:
    pass

__m = TrivialClass()
__m.nbcores = 1

def set_cores(nbcores):
    if nbcores > 0:
        __m.nbcores = nbcores
    else:
        raise RuntimeError("Cannot set a negative number of cores")

def get_cores():
    return __m.nbcores

def multiprocess(func, func_args, func_kwargs=None):
    pool = mp.Pool(processes=get_cores())
    jobs = []
    packed = pack(func)
    if func_kwargs is None:
        for a in func_args:
            jobs.append(pool.apply_async(unpack, (packed, a, None)))
    else:
        for a, b in zip(func_args, func_kwargs):
            jobs.append(pool.apply_async(unpack, (packed, a, b)))
    results = [j.get() for j in jobs]
    results.sort()
    return results

def pack(fn):
    code = marshal.dumps(fn.__code__)
    name = pickle.dumps(fn.__name__)
    defs = pickle.dumps(fn.__defaults__)
    clos = pickle.dumps(fn.__closure__)
    return (code, name, defs, clos)

def unpack(data, f_args, f_kwargs):
    code = marshal.loads(data[0])
    glob = globals()
    name = pickle.loads(data[1])
    defs = pickle.loads(data[2])
    clos = pickle.loads(data[3])
    unpacked = types.FunctionType(code, glob, name, defs, clos)
    if f_kwargs is None:
        return unpacked(*f_args)
    else:
        return unpacked(*f_args, **f_kwargs)
    return types.FunctionType(code, glob, name, defs, clos)
