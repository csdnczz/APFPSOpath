import sys
import math
import numpy as np
# from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot

plt.style.use('seaborn-pastel')

# Sent for figure
font = {'size'   : 18}
matplotlib.rc('font', **font)


x_lim = (0,100)
y_lim = (-30,30)

low = -10
high = 10

init_low_x = 40
init_high_x = 50
init_low_y = 0
init_high_y = 20


# print(rand1)


def loss(pos, data):
	global X
	totalLoss = 0
	totalLoss = np.sum(np.square(data - (pos[0]*X + pos[1])))
	return np.sqrt(totalLoss/len(data))



f  = lambda x, y: 20 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
xmin, xmax, xstep = low, high, 0.01
ymin, ymax, ystep = low, high, 0.01
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f(x, y)

lines = []
particles = []

n_particles = 10

n_iterations = 1000
interval = 30

g_value = 1000000
g_position = np.array([low + 2*high*np.random.random(), low + 2*high*np.random.random()])
error = []






def energy(pos):
	# paraboloid = pos[0]**2 + pos[1]**2
	rastrigin = 20 + pos[0]**2 + pos[1]**2 - 10*np.cos(2*np.pi*pos[0]) - 10*np.cos(2*np.pi*pos[1])
	return rastrigin

def update_velocity(particle, w0, w1, w2):
	global g_value, g_position
	new_velocity =	w0*particle.velocity + \
					w1*(particle.b_position - particle.position[-1]) + \
					w2*(g_position - particle.position[-1])

	# new_velocity =	0.9*particle.velocity + \
	# 			0.02*(g_value/(particle.b_value + g_value))*(particle.b_position - particle.position[-1]) + \
	# 			0.1*(particle.b_value/(particle.b_value + g_value))*(g_position - particle.position[-1])

	return new_velocity



class Particle:
	def __init__(self, id):
		self.id = id
		self.position = np.array([[init_low_x+ 2*init_high_x*np.random.random(), init_low_y + 2*init_high_y*np.random.random()]])
		self.velocity = 0.01*np.array([-1 + 2*np.random.random(), -1 + 2*np.random.random()])
		self.b_value = math.inf
		self.b_position = self.position[-1]

	def __str__(self):
		return "Particle {}: Position:{} Velocity:{} bVal:{} bPos{}".format(self.id,
																			self.position, 
																			self.velocity, 
																			self.b_value, 
																			self.b_position)

def generate_data(n,mu,sigma,M,C):
	rand = np.random.normal(mu,sigma,(n,))
	X = np.linspace(1,n,n)
	model = M*X + C
	return model + rand

def update(i):
	global data, g_value, g_position, error, line2, line3, lines
	print(i, g_position, g_value)
	for k in range(n_particles):	

		# Calculate target value
		f = loss(particles[k].position[-1], data)

		# Compare against pv and gv and replace if needed
		if f < particles[k].b_value:
			particles[k].b_value = f
			particles[k].b_position = particles[k].position[-1]

		if f < g_value:
			g_value = f
			g_position = particles[k].position[-1]

		# calculate new velocity
		particles[k].velocity = update_velocity(particles[k], w0, w1, w2)		

		# move
		new_particle_position = particles[k].position[-1] + particles[k].velocity

		# append new position
		particles[k].position = np.vstack([particles[k].position, new_particle_position])

		a = particles[k].position[:,0]
		b = particles[k].position[:,1]


		# print(error)

		lines[k].set_data(a,b)
	# lines[-1].set_data(time, error)
	time = np.linspace(1,i+2,i+2)
	error.append(g_value)
	# print(time, np.asarray(error))
	line2.set_data(time, np.asarray(error))

	line3.set_data(np.array([0, 500]), np.array([g_position[1], g_position[0]*500+g_position[1]]))

	return lines[0], line2, line3



if __name__ == "__main__":
	
	w0, w1, w2 = 0.9, 0.3, 0.8

	n,mu,sigma,M,C = 100,0,100,25,120

	X = np.linspace(1,n,n)
	data = generate_data(n,mu,sigma,M,C)

	fig = plt.figure(figsize=(35,15))

	# Plot 1: Contour and PSO particle animation
	ax1 = subplot2grid((2,2),(0,0),rowspan=2)
	# ax1.grid(True)
	# ax1.contour(x, y, z, levels=np.linspace(0, 200, 30), norm=LogNorm(), cmap=plt.cm.jet)
	for i in range(n_particles):
		line_i, = ax1.plot([], [], lw=1)
		lines.append(line_i)

	# Plot 2: Overall loss function plot vs time
	ax2 = subplot2grid((2,2),(0,1))
	line2, = ax2.plot([],[],lw=3)

	# Plot the linear regression in action with data and the model
	ax3 = subplot2grid((2,2),(1,1))
	line3, = ax3.plot([],[],lw=3)
	ax3.scatter(np.linspace(1,n,n), data)
	# lines.append(line2)
	# print(len(lines))
	ax1.set_xlim(0,100)
	ax1.set_ylim(0,200)
	ax1.set_xlabel("M-axis (Line Slope)")
	ax1.set_ylabel("C-axis (Line Y-Intercept)")
	

	ax2.set_xlim(0,400)
	ax2.set_ylim(0,500)
	ax2.set_xlabel("Iterations Elapsed (PSO)")
	ax2.set_ylabel("Total Residual (PSO)")


	ax3.set_xlabel("X-axis (Independent Variable)")
	ax3.set_ylabel("Y-axis (Dependent Variable)")

	ax1.grid(True)
	ax2.grid(True)
	ax3.grid(True)


	for k in range(n_particles):
		particles.append(Particle(k))


	simulation = FuncAnimation(fig, update, blit=False, frames=2000, interval=100, repeat=False)
	
	# mng = plt.get_current_fig_manager()
	# mng.resize(*mng.window.maxsize())
	plt.show()
