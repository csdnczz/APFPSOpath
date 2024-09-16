import time
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)},linewidth=np.inf)
from subprocess import call
from mayavi import mlab


goal = (20,24,3)

obstacles = [	(-15,10,5),
				(-3,16,5),
				(17,5,2),
				(1,-3,4),
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

obsScale = 1.0

def f(X,Y):
	result = (X-goal[0])**2 + (Y-goal[1])**2
	for o in obstacles:
		power = 2000*o[2]/(2.0+((X-o[0])**2 + (Y-o[1])**2)*2)
		result += power
	result -= 2000*o[2]/(2.0+((X-goal[0])**2 + (Y-goal[1])**2)*2)
	return result



xy_min, xy_max = -50, 50

u_line = np.linspace(xy_min, xy_max, 1000)
v_line = np.linspace(xy_min, xy_max, 1000)
U, V = np.meshgrid(u_line, v_line)

G = f(U,V)

sWN = mlab.mesh(U, V, G/100)
mlab.show()