
from functions import *
import numpy as np

x=np.linspace(0,100,101)
f1 = lambda x: np.ones(x.size)
f2 = lambda x: x
f3 = lambda x: x**2
y = 3*f1(x) + 2*f2(x) + 4*f3(x) + 3*(np.random.random()-.5)
print(general_regression(y=y,F=np.vstack([f1(x),f2(x),f3(x)]).T))
