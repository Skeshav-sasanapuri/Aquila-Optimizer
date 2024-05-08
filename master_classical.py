from mpire import WorkerPool
from worker_AO_classical import worker
from itertools import product

if __name__ == "__main__":
    functions = [f"F{i}" for i in range(1, 14)]
    dims = [10, 50, 100, 500]
    # os.chdir("D:\Bio inspired\project")
    # for f in functions:
    f = {
        "F14": dict(LB=-65, UB=65, Dim=2),
        "F15": dict(LB=-5, UB=5, Dim=4),
        "F16": dict(LB=-5, UB=5, Dim=2),
        "F17": dict(LB=-5, UB=5, Dim=2),
        "F18": dict(LB=-2, UB=2, Dim=2),
        "F19": dict(LB=0, UB=1, Dim=3),
        "F20": dict(LB=0, UB=1, Dim=6),
        "F21": dict(LB=0, UB=10, Dim=4),
        "F22": dict(LB=0, UB=10, Dim=4),
        "F23": dict(LB=0, UB=10, Dim=4),
    }
    functions2 = [(k, f[k]["Dim"]) for k in f.keys()]
    with WorkerPool(n_jobs=6, enable_insights=True) as pool:
        results = pool.map(
            worker, list(product(functions, dims)) + functions2, progress_bar=True
        )
    # with WorkerPool(n_jobs=6, enable_insights=True) as pool:
    #     results = pool.map(worker, product(functions, dims), progress_bar=True)
