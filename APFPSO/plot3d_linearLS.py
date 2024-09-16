from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


x_line = np.linspace(-10000, 10000, 30)
y_line = np.linspace(-10000, 10000, 30)

meshX, meshY = np.meshgrid(x_line, y_line)

n = 100

mu = 0
sigma = 200

M = 50
C = 20

rand0 = np.random.normal(mu,sigma,(n,))
rand1 = 10*np.random.random(n)
X = np.linspace(1,n,n)
line = M*X + C
data = line + rand0
print(rand1)

def loss(a,b):
	global data
	totalLoss = 0
	totalLoss = np.sum(np.square(data - (a*X + b)))
	return totalLoss/len(data)


# Z = function(X,Y)

fig = plt.figure(figsize=(35,15))

# Plot 1: 3D view of the loss function
ax = fig.add_subplot(1,2,1,projection="3d")

zs = np.array([loss(x,y) for x,y in zip(np.ravel(meshX), np.ravel(meshY))])
Z = zs.reshape(meshX.shape)

ax.plot_surface(meshX, meshY, Z, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')

# Plotting the linear regression and data on XY
ax = fig.add_subplot(1,2,2)

ax.plot(X,data,'bo')
ax.grid(True)

ax.set_xlabel("M-axis")
ax.set_ylabel("C-axis")

# ax.set_zlabel("Total Residual")



plt.xlabel('X-axis', fontsize=24, color='blue')
plt.ylabel('Y-axis', fontsize=24, color='blue')
plt.show()