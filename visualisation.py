import matplotlib.pyplot as plt
import numpy as np

temperatures=np.genfromtxt("pipe_0_output.csv", delimiter=",", dtype=np.float64)


fig, ax=plt.subplots(figsize=(5,5))
ax.imshow(temperatures, aspect="auto")
# ax.colorbar(label='Valeurs')
fig.show()
# print(temperatures)
# plt.plot(temperatures[])