# Aquila-Optimizer
This is a team project for our class CSCI 633: Biologically Inspired Intelligent systems. We have implemented the Aquila Optimizer algorithm as a reproducability study of the original paper: 


Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A.A., Al-Qaness, M.A. and Gandomi, A.H., 2021. Aquila optimizer: a novel meta-heuristic optimization algorithm. Computers & Industrial Engineering, 157, p.107250.


## Intstructions

### Instructions to run optimizer for custom objective functions
In order to use the optimizer on a custom objective function or test it out with a few simple objective functions, please use the aquila_optimizer.py file.
This file includes the implementation of the aquila optimizer algorithm with a few objective functions written by us. A custom objective function could be written and passed to the 
optimizer function call on line 161: 

Best_FF, Best_P, conv = AO(population_size, M_Iter, LB_fourpeak, UB_fourpeak, Dim, four_peak) 

along with the custom upper and lower bounds in the form of numpy arrays.

example implementation of a custom objective function:
def eggcrate(x):
    t1 = x[0]**2
    t2 = x[1]**2
    t3 = 25 * ((np.sin(x[0])**2) + np.sin(x[1])**2)
    return (t1 + t2 + t3)

example format of upper and lower bounds to be passed to the optmizer funcion call:
LB_eggcrate = np.array([-np.pi, -np.pi])

UB_eggcrate = np.array([np.pi, np.pi])

### Instructions to run tests 
The code to run 29 CEC 2017 benchmark functions, 9 CEC 2019 benchmark functions and 23 classical benchmark functions have been provided. 
Running the master.py would run the CEC benchmark functions tests and running the master_classical.py would run the classical benchmark function tests


