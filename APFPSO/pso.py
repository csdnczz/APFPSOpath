import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
plt.style.use('seaborn-pastel')

low = -10
high = 10

init_low_x = -10
init_high_x = 10
init_low_y = -10
init_high_y = 10


f  = lambda x, y: 20 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
xmin, xmax, xstep = low-10, high-10, 0.01
ymin, ymax, ystep = low+20, high+20, 0.01
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f(x, y)

lines = []
particles = []

n_particles = 10

n_iterations = 500
interval = 100

g_value = 1000000
g_position = np.array([low + 2*high*np.random.random(), low + 2*high*np.random.random()])
best = (0.9, 0.11, 0.8)
demo = (0.95, 0.001, 0.05)
w0, w1, w2 = demo


if len(sys.argv) > 1:
	if sys.argv[1] == "-visual":
		fig = plt.figure(figsize=(35,15))
		ax = plt.axes(xlim=(low-10, high-10), ylim=(low-20, high-20))
		ax.contour(x, y, z, levels=np.linspace(0, 200, 30), norm=LogNorm(), cmap=plt.cm.jet)
		
		for i in range(n_particles):
			line_i, = ax.plot([], [], lw=4)
			lines.append(line_i)


def energy(pos):
	# paraboloid = pos[0]**2 + pos[1]**2
	x = pos[0] + 10
	y = pos[1] + 20
	rastrigin = 20 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
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

def init():
    return tuple(lines)

def animate(i, fargs):
	global g_value, g_position
	print(i, g_position, g_value)
	for k in range(n_particles):	

		# Calculate target value
		f = energy(particles[k].position[-1])

		# Compare against pv and gv and replace if needed
		if f < particles[k].b_value:
			particles[k].b_value = f
			particles[k].b_position = particles[k].position[-1]

		if f < g_value:
			g_value = f
			g_position = particles[k].position[-1]

		# calculate new velocity
		particles[k].velocity = update_velocity(particles[k], w0, fargs[0], fargs[1])		

		# move
		new_particle_position = particles[k].position[-1] + particles[k].velocity

		# append new position
		particles[k].position = np.vstack([particles[k].position, new_particle_position])

		a = particles[k].position[:,0]
		b = particles[k].position[:,1]

		if len(sys.argv) > 1 and sys.argv[1] == "-visual":
			lines[k].set_data(a,b)
	
	# print("Iterations:", i+1, "G_best:", g_value, "G_pos:", g_position, "W:", fargs)

	if i+1 == n_iterations:
		sys.exit()

	return tuple(lines)


class Particle:
	def __init__(self, id):
		self.id = id
		self.position = np.array([[init_low_x+ 2*init_high_x*np.random.random(), init_low_y + 2*init_high_y*np.random.random()]])
		self.velocity = np.array([2*np.random.random(), 2*np.random.random()])
		self.b_value = math.inf
		self.b_position = self.position[-1]

	def __str__(self):
		return "Particle {}: Position:{} Velocity:{} bVal:{} bPos{}".format(self.id,
																			self.position, 
																			self.velocity, 
																			self.b_value, 
																			self.b_position)




if __name__ == "__main__":

	for k in range(n_particles):
		particles.append(Particle(k))

	j = 0

	if len(sys.argv) > 1:
		if sys.argv[1] == "-visual":
			anim = FuncAnimation(fig, animate, init_func=init,
		                               frames=n_iterations, interval=interval, blit=True, fargs=((w1,w2),))

			
			plt.show()

	else:
		g_min = 100000
		g_min_weights = (0,0)
		f = open("meta_function.txt", "w")
		for a in range(100):
			for b in range(100):
				g_avg = 0
				for trial in range(20):
					particles = []
					for k in range(n_particles):
						particles.append(Particle(k))

					g_value = 1000000
					g_position = np.array([low + 2*high*np.random.random(), low + 2*high*np.random.random()])

					for j in range(n_iterations):
						animate(j, (a*0.01,b*0.01))
						j+=1

					g_avg += g_value
				g_avg /= 10
				if g_avg < g_min:
					g_min = g_avg
					g_min_weights = a*0.01, b*0.01
					# print("Trial:", trial, "G_best:", g_value, "G_pos:", g_position, "W:", (a,b))
				print("Trial:", trial, "Iterations:", n_iterations, "G_avg:", g_avg, "G_min", g_min, "W:", (a*0.01,b*0.01), "MinW:",g_min_weights)
				f.write("{},{},{}\n".format(a*0.01, b*0.01, g_avg))
