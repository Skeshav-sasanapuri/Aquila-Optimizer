import opfunu
import numpy as np
from mealpy import FloatVar
from aquila_optimizer import AO
from sys import argv
from json import dump


def run_trials_AO(f, trials=30):
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
            y, x, conv = AO(30, 500, f.lb, f.ub, 10, f.evaluate)
            sols.append(y)
            iters.append(conv)
        except Exception as e:
            print(f"error {e} in AO and {f}")
    # return {
    #     "mean": sols.mean(),
    #     "std": sols.std(),
    #     "worst": sols.max(),
    #     "best": sols.min(),
    #     "convergence": iters[sols.argmin()],
    # }
    sols = np.array(sols)
    if len(sols) == 0:
        dump(
            {"trials": 0},
            open(f"./outputs/{f.__class__.__name__}_AO.json", "w"),
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
            open(f"./outputs/{f.__class__.__name__}_AO.json", "w"),
        )


def worker(fn_name):
    # print(f"Started working on {fn_name}")

    DIM = 10
    f = opfunu.get_functions_by_classname(fn_name)[0](DIM)
    op = []
    op.append(run_trials_AO(f))


if __name__ == "__main__":
    worker(argv[1])
