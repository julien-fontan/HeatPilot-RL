import matplotlib.pyplot as plt
import numpy as np

# temperatures=np.genfromtxt("pipe_0_output.csv", delimiter=",", dtype=np.float64)


# fig, ax=plt.subplots(figsize=(5,5))
# ax.imshow(temperatures, aspect="auto")
# # ax.colorbar(label='Valeurs')
# fig.show()
# # print(temperatures)
# # plt.plot(temperatures[])

mass_flows=np.genfromtxt("mass_flows_final.csv")
velocities=np.genfromtxt("velocities_final.csv")

fig2, ax2 = plt.subplots(figsize=(10, 5))
x_values = np.arange(1, len(mass_flows) + 1)
ax2.plot(x_values, mass_flows)
ax2.set_title("Mass Flows per Row")
ax2.set_xticks(x_values)
ax2.legend()
fig2.show()