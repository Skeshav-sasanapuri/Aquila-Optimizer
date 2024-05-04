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
            F_UB = X[i, :] > UB
            F_LB = X[i, :] < LB
            X[i, :] = (X[i, :] * ~(F_UB + F_LB)) + UB * F_UB + LB * F_LB
            Ffun[i] = F_obj(X[i, :])
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

        if t % 100 == 0:
            print('At iteration', t, 'the best solution fitness is', Best_FF)
        conv.append(Best_FF)
        t += 1

    return Best_FF, Best_P, conv


# Example usage
Solution_no = 20
F_name = 'F8'
M_Iter = 1000
LB = np.array([-5, -5])
UB = np.array([5, 5])
Dim = 2


def F_obj(x):
    # Define your objective function here
    return x[0] ** 2 + x[1] ** 2


Best_FF, Best_P, conv = AO(Solution_no, M_Iter, LB, UB, Dim, F_obj)

print('The best-obtained solution by AO is:', Best_P)
print('The best optimal values of the objective function found by AO is:', Best_FF)
