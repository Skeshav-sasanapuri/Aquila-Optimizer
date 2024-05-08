import opfunu
from mpire import WorkerPool
from worker_AO import worker

if __name__ == "__main__":
    functions = [f.__name__ for f in opfunu.get_functions_based_classname("2017")]
    functions += [f.__name__ for f in opfunu.get_functions_based_classname("2019")]
    # os.chdir("D:\Bio inspired\project")
    # for f in functions:
    with WorkerPool(n_jobs=6, enable_insights=True) as pool:
        results = pool.map(worker, functions, progress_bar=True)
