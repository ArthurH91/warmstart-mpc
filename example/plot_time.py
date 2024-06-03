import matplotlib.pylab as plt
import numpy as np

a = np.load("/home/arthur/Desktop/Code/warmstart-mpc/example/time_result__1-06-24.npy", allow_pickle=True)
print(a)
plt.plot(a)
plt.show()