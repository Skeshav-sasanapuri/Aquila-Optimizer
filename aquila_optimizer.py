import numpy as np


def initialization(N, Dim, UB, LB):
    B_no = len(UB)
    X = np.zeros((N, Dim))

    if B_no == 1:
        X = np.random.rand(N, Dim) * (UB - LB) + LB
    elif B_no > 1:
        for i in range(Dim):
            Ub_i = UB[i]
            Lb_i = LB[i]
            X[:, i] = np.random.rand(N) * (Ub_i - Lb_i) + Lb_i

    return X


def Levy(d):
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
            np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return step


def AO(N, T, LB, UB, Dim, F_obj):
    Best_P = np.zeros(Dim)
    Best_FF = float('inf')
    X = initialization(N, Dim, UB, LB)
    X_new = X.copy()
    Ffun = np.zeros(N)
    Ffun_new = np.zeros(N)
    t = 1
    alpha = 0.1
    delta = 0.1
    conv = []

    while t < T + 1:
        for i in range(N):
            # F_UB = X[i, :] > UB
            # F_LB = X[i, :] < LB
            # X[i, :] = (X[i, :] * ~(F_UB + F_LB)) + UB * F_UB + LB * F_LB
            x_val = X[i, :]
            f_val = F_obj(x_val)
            Ffun[i] = f_val
            if Ffun[i] < Best_FF:
                Best_FF = Ffun[i]
                Best_P = X[i, :]

        G2 = 2 * np.random.rand() - 1
        G1 = 2 * (1 - t / T)
        to = np.arange(1, Dim + 1)
        u = 0.0265
        r0 = 10
        r = r0 + u * to
        omega = 0.005
        phi0 = 3 * np.pi / 2
        phi = -omega * to + phi0
        x = r * np.sin(phi)
        y = r * np.cos(phi)
        QF = t ** ((2 * np.random.rand() - 1) / (1 - T) ** 2)

        for i in range(N):
            if t <= (2 / 3) * T:
                if np.random.rand() < 0.5:
                    X_new[i, :] = Best_P * (1 - t / T) + (np.mean(X[i, :]) - Best_P) * np.random.rand()
                    Ffun_new[i] = F_obj(X_new[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = X_new[i, :]
                        Ffun[i] = Ffun_new[i]
                else:
                    X_new[i, :] = Best_P * Levy(Dim) + X[np.random.randint(N), :] + (y - x) * np.random.rand()
                    Ffun_new[i] = F_obj(X_new[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = X_new[i, :]
                        Ffun[i] = Ffun_new[i]
            else:
                if np.random.rand() < 0.5:
                    X_new[i, :] = (Best_P - np.mean(X)) * alpha - np.random.rand() + (
                            (UB - LB) * np.random.rand() + LB) * delta
                    Ffun_new[i] = F_obj(X_new[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = X_new[i, :]
                        Ffun[i] = Ffun_new[i]
                else:
                    X_new[i, :] = QF * Best_P - G2 * X[i, :] * np.random.rand() - G1 * Levy(Dim) + np.random.rand() * G2
                    Ffun_new[i] = F_obj(X_new[i, :])
                    if Ffun_new[i] < Ffun[i]:
                        X[i, :] = X_new[i, :]
                        Ffun[i] = Ffun_new[i]

        if t % 1 == 0:
            print('At iteration', t, 'the best solution fitness is {:.8f}'.format(Best_FF))
        conv.append(Best_FF)
        t += 1

    return Best_FF, Best_P, conv


# Example usage
population_size = 100
F_name = 'F8'
M_Iter = 100
Dim = 2

# Bounds
LB_rosenbrock = np.array([-5, -5])
UB_rosenbrock = np.array([5, 5])
LB_alpine = np.array([-10, -10])
UB_alpine = np.array([10, 10])
LB_ackleys = np.array([-10, -10])
UB_ackleys = np.array([10, 10])
LB_easom = np.array([-100, -100])
UB_easom = np.array([100, 100])
LB_eggcrate = np.array([-np.pi, -np.pi])
UB_eggcrate = np.array([np.pi, np.pi])
LB_fourpeak = np.array([-5, -5])
UB_fourpeak = np.array([5, 5])


def F_obj(x):
    t1 = np.exp(-((x[0] - 4) ** 2) - (x[1] - 4) ** 2)
    t2 = np.exp(-((x[0] + 4) ** 2) - (x[1] - 4) ** 2)
    t3 = 2 * (np.exp(-(x[0] ** 2) - (x[1] ** 2)) + np.exp(-(x[0] ** 2) - ((x[1] + 4) ** 2)))
    return t1 + t2 + t3


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def alpine(x):
    return np.sum(np.abs((x * np.sin(x)) + (0.1 * x)))


def ackleys(x):  # returns a scalar
    term1 = -20 * np.exp(-0.02 * np.sqrt((1 / 2) * np.sum(x ** 2)))
    term2 = -np.exp((1 / 2) * np.sum(np.cos(2 * np.pi * x)))
    return term1 + term2 + 20 + np.exp(1)

def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2) - ((x[1] - np.pi) ** 2))

def eggcrate(x):
    t1 = x[0]**2
    t2 = x[1]**2
    t3 = 25 * ((np.sin(x[0])**2) + np.sin(x[1])**2)
    return (t1 + t2 + t3)

def four_peak(x):
    t1 = np.exp(-((x[0] - 4) ** 2) - (x[1] - 4) ** 2)
    t2 = np.exp(-((x[0] + 4) ** 2) - (x[1] - 4) ** 2)
    t3 = 2 * (np.exp(-(x[0] ** 2) - (x[1] ** 2)) + np.exp(-(x[0] ** 2) - ((x[1] + 4) ** 2)))
    return (t1 + t2 + t3) * -1


print("fourpeak function: ", four_peak(np.array([0, 0])))
Best_FF, Best_P, conv = AO(population_size, M_Iter, LB_fourpeak, UB_fourpeak, Dim, four_peak)

print('The best-obtained solution by AO is: [{:.8f}'.format(Best_P[0]), '{:.8f}'.format(Best_P[1]), ']')
print('The best optimal values of the objective function found by AO is: {:.8f}'.format(Best_FF))


Max_iter = 100
pop_size = 30
G2 = 2 * np.random.rand() - 1
G1 = 2 * (1 - t / T)
to = np.arange(1, Dim + 1)
u = 0.0265
r0 = 10
r = r0 + u * to
omega = 0.005
phi0 = 3 * np.pi / 2
phi = -omega * to + phi0
x = r * np.sin(phi)
y = r * np.cos(phi)
beta = 1.5
s = 0.5
alpha = 0.1
delta = 0.1
