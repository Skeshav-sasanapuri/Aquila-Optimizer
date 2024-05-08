import numpy as np
from aquila_optimizer import AO
from sys import argv
from json import dump
import matlab.engine
from mealpy import GOA, EO, PSO, ALO, GWO, MPA, SSA, SCA, WOA, SMA
from mealpy import FloatVar


class matlab_function:
    metadata = {
        "F1": dict(LB=-100, UB=100, Dim=10),
        "F2": dict(LB=-10, UB=10, Dim=10),
        "F3": dict(LB=-100, UB=100, Dim=10),
        "F4": dict(LB=-100, UB=100, Dim=10),
        "F5": dict(LB=-30, UB=30, Dim=10),
        "F6": dict(LB=-100, UB=100, Dim=10),
        "F7": dict(LB=-1.28, UB=1.28, Dim=10),
        "F8": dict(LB=-500, UB=500, Dim=10),
        "F9": dict(LB=-5.12, UB=5.12, Dim=10),
        "F10": dict(LB=-32, UB=32, Dim=10),
        "F11": dict(LB=-600, UB=600, Dim=10),
        "F12": dict(LB=-50, UB=50, Dim=10),
        "F13": dict(LB=-50, UB=50, Dim=10),
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

    def __init__(self, eng, f, dim=None):
        self.f = f
        self.eng = eng
        if dim:
            self.dim = dim
        else:
            self.dim = self.metadata[f]["Dim"]
        self.lb = np.repeat(self.metadata[f]["LB"], self.dim)
        self.ub = np.repeat(self.metadata[f]["UB"], self.dim)

    def evaluate(self, x):
        return getattr(self.eng, self.f)(matlab.double(x.tolist()))


def run_trials(f, m, trials=30, dim=None):
    problem_dict = {
        "obj_func": f.evaluate,
        "bounds": FloatVar(lb=f.lb, ub=f.ub),
        "minmax": "min",
        "log_to": None,
    }
    sols = []
    iters = []
    for i in range(trials):
        try:
            m.solve(problem_dict)
            sols.append(f.evaluate(m.g_best.solution))
            iters.append(m.history.list_global_best_fit)
        except Exception as e:
            print(f"error {e} in {m} and {f.f}")
    sols = np.array(sols)
    if len(sols) == 0:
        dump(
            {"trials": 0},
            open(f"./outputs/Classical{f.f}_{m.__class__.__name__}_{f.dim}.json", "w"),
        )
    else:
        dump(
            {
                "trials": len(sols),
                "mean": sols.mean(),
                "std": sols.std(),
                "worst": sols.max(),
                "best": sols.min(),
                "convergence": iters[sols.argmin()],
            },
            open(f"./outputs/Classical{f.f}_{m.__class__.__name__}_{f.dim}.json", "w"),
        )


def run_trials_AO_classical(f, trials=30, dim=10):
    sols = []
    iters = []
    for i in range(trials):
        # try:
        y, x, conv = AO(30, 100, f.lb, f.ub, dim, f.evaluate)
        sols.append(y)
        iters.append(conv)
        # except Exception as e:
        #     print(f"error {e} in AO and {f}")
    sols = np.array(sols)
    if len(sols) == 0:
        op = {"trials": 0}
    else:
        op = {
            "trials": len(sols),
            "mean": sols.mean(),
            "std": sols.std(),
            "worst": sols.max(),
            "best": sols.min(),
            "convergence": iters[sols.argmin()],
        }
    dump(
        op,
        open(f"./outputs/Classical{f.f}_AO_{dim}.json", "w"),
    )


def worker(fn_name, dim=None):
    # print(f"Started working on {fn_name}")

    eng = matlab.engine.start_matlab()
    eng.addpath(r"C:\Users\aakas\Downloads\Compressed\AO\AO")
    f = matlab_function(eng, fn_name, dim)
    models = [
        GOA.OriginalGOA(epoch=100, pop_size=30),
        EO.OriginalEO(epoch=100, pop_size=30),
        PSO.OriginalPSO(epoch=100, pop_size=30, c1=2, c2=2),
        # DA implementation dosent exist in this library
        ALO.OriginalALO(epoch=100, pop_size=30),
        GWO.OriginalGWO(epoch=100, pop_size=30),
        MPA.OriginalMPA(epoch=100, pop_size=30),
        SSA.OriginalSSA(epoch=100, pop_size=30),
        SCA.OriginalSCA(epoch=100, pop_size=30),
        WOA.OriginalWOA(epoch=100, pop_size=30),
        SMA.OriginalSMA(epoch=100, pop_size=30),
    ]
    for m in models:
        run_trials(f, m, trials=15)

    run_trials_AO_classical(f, trials=15, dim=f.dim)
    eng.quit()


if __name__ == "__main__":
    worker(argv[1], 10)
