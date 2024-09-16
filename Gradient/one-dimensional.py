# Uncomment the next two lines if you want to save the animation
#import matplotlib
#matplotlib.use("Agg")
from functions import *
import numpy as np
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation



# Sent for figure
font = {'size'   : 18}
matplotlib.rc('font', **font)

# Setup figure and subplots
f0 = figure(num = 0, figsize = (35, 16))#, dpi = 100)
f0.suptitle("Gradient Descent", fontsize=12)

x1_low, x1_high, y1_low, y1_high, x1_step = -8, 8, -2, 1000, 0.01
x2_low, x2_high, y2_low, y2_high, x2_step = 0, 200, 0, 20, 0.1

ax01 = subplot2grid((1, 2), (0, 0))
ax01.set_xlim(x1_low, x1_high)
ax01.set_ylim(y1_low, y1_high)
ax01.grid(True)
ax01.set_title('Gradient Descent on f(x)')
ax01.set_xlabel("X-axis")
ax01.set_ylabel("Function")

ax02 = subplot2grid((1, 2), (0, 1))
ax02.set_xlim(x2_low, x2_high)
ax02.set_ylim(y2_low, y2_high)
ax02.grid(True)
ax02.set_title('Cost vs Time')
ax02.set_xlabel("Iterations")
ax02.set_ylabel("Function Minimum")


# Data Placeholders
hist=zeros(0)
cost=zeros(0)

m=zeros(0)
del_x = 0.01
del_grad_x = 0.01

def grad(f, x):
	return (f(x + del_grad_x) - f(x - del_grad_x))/(2*del_grad_x)

def update_x(f,x):
	g = grad(f,x)
	upd = -g*del_x
	print("Grad:", g, "-g*dx:", upd)
	result = x + upd
	return result


def calc_cost(x):
	return biquad(x)

def updateData(self):
	global x, m, hist, cost

	value = calc_cost(x)
	ax01.plot([x],[value], 'ro', markersize=3)
	# print("Iteration:",len(t),len(hist), "Position:", x, "Cost:", value)
	hist = append(hist,value)
	cost = append(cost, value)
	m = append(m,x)



	x = update_x(calc_cost, x)




	line = np.arange(x1_low, x1_high, x1_step)
	graph = calc_cost(line)

	t = np.arange(0, len(cost), 1)

	p00.set_data(line, graph)
	p01.set_data(m, hist)
	p10.set_data(t,cost)

	# if x >= xmax-1.00:
	# 	p00.axes.set_xlim(x-xmax+1.0,x+1.0)
	# 	p10.axes.set_xlim(x-xmax+1.0,x+1.0)


	return p00, p10, p01


if __name__ == "__main__":
	global x

	# set plots
	p00, = ax01.plot(m,hist,'r-', label="Gradient Descent")
	p01, = ax01.plot([],[],lw=4, color='b')
	p10, = ax02.plot(m,cost,'b-', label="Cost")



	# set lagends
	ax01.legend([p00], [p00.get_label()])
	ax02.legend([p10], [p10.get_label()])

	# Data Update
	xmin = -10.0
	xmax = 10.0
	x = 4


	simulation = animation.FuncAnimation(f0, updateData, blit=False, frames=2000, interval=1, repeat=False)

	plt.show()
