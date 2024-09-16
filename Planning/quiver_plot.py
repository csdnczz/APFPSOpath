import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,2*np.pi+2*np.pi/20,2*np.pi/30)
y = np.arange(0,2*np.pi+2*np.pi/20,2*np.pi/30)

X,Y = np.meshgrid(x,y)

u = np.sin(X) + np.sin(Y)
v = -np.sin(X) + np.sin(Y)

fig, ax = plt.subplots(figsize=(7,7))

ax.quiver(X,Y,u,v)

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.axis([0,2*np.pi,0,2*np.pi])
ax.set_aspect('equal')

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()