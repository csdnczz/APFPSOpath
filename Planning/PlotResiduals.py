import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import *

a_star = np.loadtxt('Residuals/Final/AStar_Residual.csv')/4
dijkstra = np.loadtxt('Residuals/Final/Dijkstra_Residual.csv')/4

apf_pso = np.loadtxt('Residuals/Final/APF_PSO_Residual.csv')
apf_pso_n10 = np.loadtxt('Residuals/Final/APF_PSO_Residual_n10.csv')
apf_pso_n20 = np.loadtxt('Residuals/Final/APF_PSO_Residual_n20.csv')
apf_pso_n30 = np.loadtxt('Residuals/Final/APF_PSO_Residual_n30.csv')

print(a_star.shape, dijkstra.shape, apf_pso.shape)

plt.style.use('seaborn-pastel')
font = {'size': 28}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(30,18))

plt.plot(a_star[0:200], 'b--', lw=8, label='A-Star Search')
plt.plot(dijkstra[0:200], 'r--', lw=8, label="Dijkstra's Algorithm")
plt.plot(apf_pso[0:200], 'g-', lw=8, label='APF-PSO (n=3)')
plt.plot(apf_pso_n10[0:200], 'y-', lw=8, label='APF-PSO (n=10)')
plt.plot(apf_pso_n20[0:200], 'v-', lw=8, label='APF-PSO (n=20)')
plt.plot(apf_pso_n30[0:200], 'o-', lw=8, label='APF-PSO (n=30)')
plt.legend()
plt.grid(True)

plt.xlabel('Number of Iterations')
plt.ylabel('Total Energy of the System (Lower is Better)')

extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('Residuals/Final/Combined_Residual_Plot.png', bbox_inches=extent.expanded(0.9, 0.9))

plt.show()

