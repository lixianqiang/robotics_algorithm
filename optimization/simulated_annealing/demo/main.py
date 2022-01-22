from optimization.simulated_annealing.script.simulated_annealing import *
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':

    def CostFunc(x):
        y = (x**2 - 5*x) * np.sin(x**2)
        return y

    def SolutionGenerator(x):
        x = np.random.uniform(-10, 10)
        return x

    def Temperature(T):
        """退火方案1"""
        iter, max_iter = 0, 5000
        k = iter
        k_max = max_iter
        T = T * (1 - (k + 1) / k_max)
        return T

    def bound():
        return [-10, 10]

    def isSafity(t):
        pass

    def Temperature2(T):
        """退火方案2"""
        proportion = 0.9
        T = proportion * T
        return T

    def isSafity2(t):
        if t < 1e-6:
            return True
        else:
            return False

    optimizedObject = {'objectFunction': CostFunc,
                       'x0': 6,
                       'constraints': SolutionGenerator,
                       'bounds': bound}

    coolingSchedule = {'temperatureFunction': Temperature,
                       'initialTemperature': 2000,
                       'terminatorCondition': isSafity2}

    sa = SimulatedAnnealing(optimizedObject, coolingSchedule, 0.01)
    res = sa.solved()

    T = [i for i in np.arange(-10, 10, 0.1)]
    X = [CostFunc(x) for x in T]
    plt.plot(T, X)
    plt.scatter(res, CostFunc(res), c='r')
    plt.show()