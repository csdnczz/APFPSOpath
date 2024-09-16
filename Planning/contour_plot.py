from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9,9))
ax = plt.axes(projection="3d")

goal = (20,24,3)

obstacles = [	(-15,10,5),
				(-3,16,5),
				(17,5,2),
				# (1,-3,4),
				(0,-20,6),
				(20,12,4),
				(-14,-18,4),
				(13,-12,4),
				(-10,-8,4),
				(14,-20,2),
				(8,5,4),
				(-22,-3,4),
				(10,20,3),
				(23,-2,4)
			]


x_line = np.linspace(-50, 50, 80)
y_line = np.linspace(-50, 50, 80)

X, Y = np.meshgrid(x_line, y_line)


# Z = function(X,Y)

Z = (X-goal[0])**2 + (Y-goal[1])**2 + 1/(1+X**2) + 1/(1+Y**2) - 100
for o in obstacles:
	power = 2000*o[2]/(0.8+((X-o[0])**2 + (Y-o[1])**2)*2)
	Z += power

# Z -= 2000/(0.01 + X**2 + Y**2)

# ax.plot_wireframe(X, Y, Z, color='green')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')

plt.show()