import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import statsmodels.api as sm

x, y, w = make_regression(
    n_samples=50, n_features=1, bias=100, noise=10, random_state=0, coef=True
)

xx = np.linspace(-3, 3, 100)
yy = w * xx + 100

def make_regression2(n_samples, bias, noise, random_state):
    np.random.seed(random_state)

    w = np.random.uniform(0,20)
    xs = np.random.uniform(-5, 5, n_samples)

    epsilons_ = np.random.randn(n_samples)
    epsilons = epsilons_ * noise

    ys = w * xs + bias + epsilons
    
    return xs, ys, w
        
x_, y_, w_ = make_regression2(n_samples=50, bias=100, noise=10, random_state=0)

xx_ = np.linspace(-5, 5, 100)
yy_ = w_ * xx_ + 100

_, axs = plt.subplots(1, 2)

axs[0].plot(xx, yy, "r-", color="blue")
axs[0].scatter(x, y, s=10, color="red")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

axs[1].plot(xx_, yy_, "r-", color="blue")
axs[1].scatter(x_, y_, s=10, color="red")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

plt.title("make_regression")
plt.show()
