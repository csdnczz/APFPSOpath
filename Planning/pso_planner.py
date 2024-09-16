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
font = {'size'   : 8}
matplotlib.rc('font', **font)


x_lim = (-30,30)
y_lim = (-30,30)

low = -100
high = 100

init_low_x = -20
init_high_x = -19
init_low_y = -20
init_high_y = -19


# print(rand1)
n_particles = 5

n_iterations = 300
interval = 0

start = (-30,30,2)	
goal = (30,-30,3)
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
obs_np = list(map(np.array,obstacles))

goal_reached = False


pdots_x = [0 for i in range(n_particles)]
pdots_y = [0 for i in range(n_particles)]
lines = []
guides = []
particles = []


g_value = 1000000
g_position = np.array([low + 2*high*np.random.random(), low + 2*high*np.random.random()])
error = []



def computeField(particle):
	p_pos = particle.position[-1]
	x = p_pos[0]
	y = p_pos[1]
	energy = (np.square(x-goal[0]) + np.square(y-goal[1]))*200
	for o in obstacles:
		energy -= (np.square(x-o[0]) + np.square(y-o[1]))*2
	return energy


def update_velocity(particle, w0, w1, w2):
	global g_value, g_position
	for o in obs_np:
		distance = np.linalg.norm(particle.position[-1] - np.array([o[:2]]))
		r = o[2] + 1.5
		if distance < r:
			new_velocity = (particle.position[-1] - np.array([o[:2]]) + np.random.normal(size=(1,2)) )/3
			return new_velocity

		new_velocity =	w0*particle.velocity + \
						w1*(particle.b_position - particle.position[-1]) + \
						w2*(g_position - particle.position[-1])

	return new_velocity

def update_position(particle):
	p_vel = particle.velocity
	new_position = particle.position[-1] + p_vel
	if new_position.shape[0]==1:
		new_position = new_position[0]
	if p_vel.shape[0]==1:
		p_vel = p_vel[0]
	if new_position[0] < x_lim[0]:
		new_position[0] = x_lim[0]
		p_vel[0] *= -1
	if new_position[0] > x_lim[1]:
		new_position[0] = x_lim[1]
		p_vel[0] *= -1
	if new_position[1] < y_lim[0]:
		new_position[1] = y_lim[0]
		p_vel[1] *= -1
	if new_position[1] > y_lim[1]:
		new_position[1] = y_lim[1]
		p_vel[1] *= -1
	return new_position


def goal_check(particle):
	distance = np.linalg.norm(particle.position[-1] - np.array([goal[:2]]))
	if distance < goal[2]:
		return True
	else:
		return False

class Particle:
	def __init__(self, id):
		self.id = id
		self.position = np.array([[start[0], start[1]]])
		self.velocity = np.array([np.abs(np.random.random()), np.abs(np.random.random())])
		self.b_value = math.inf
		self.b_position = self.position[-1]

	def __str__(self):
		return "Particle {}: Position:{} Velocity:{} bVal:{} bPos{}".format(self.id,self.position,self.velocity,self.b_value,self.b_position)


def simulate(w0,w1,w2):
	global g_value, g_position, goal_reached
	particle_id = -1
	while not(goal_reached):
		for k in range(n_particles):	

			f = computeField(particles[k])
			if f < particles[k].b_value:
				particles[k].b_value = f
				particles[k].b_position = particles[k].position[-1]

			if f < g_value:
				g_value = f
				g_position = particles[k].position[-1]

			particles[k].velocity = update_velocity(particles[k], w0, w1, w2)	
			new_particle_position = update_position(particles[k])
			particles[k].position = np.vstack([particles[k].position, new_particle_position])
			if goal_check(particles[k]):
				particle_id = k
				goal_reached = True
				break

			a = particles[k].position[:,0]
			b = particles[k].position[:,1]

		print(g_position)
	return particle_id


if __name__ == "__main__":
	global fig
	best = (0.9, 0.11, 0.8)
	demo = (0.95, 0.001, 0.05)
	w0, w1, w2 = demo
	m = 9
	for k in range(n_particles):
		particles.append(Particle(k))

	fig = plt.figure(figsize=(16,17))
	ax1 = subplot2grid((1,1),(0,0))


	x = np.arange(-30,30,2)
	y = np.arange(-30,30,2)
	X,Y = np.meshgrid(x,y)
	u = (-(X-goal[0]))
	v = (-(Y-goal[1]))

	ax1.quiver(X,Y,u,v)

	for o in obstacles:
		circle = plt.Circle((o[0], o[1]), o[2], color='r')
		ax1.add_artist(circle)



	p_id = simulate(w0,w1,w2)

	p = particles[p_id]


	clrs = ['y','r','g']
	for j in range(len(particles)):
		if j != p_id:
			for i in range(1,len(particles[j].position)):
				a = particles[j].position[i]
				b = particles[j].position[i-1]
				xs,ys = [a[0],b[0]],[a[1],b[1]]
				plt.plot(xs,ys,color=clrs[j%len(clrs)],lw=1)
			plt.plot(particles[j].position[-1,0],particles[j].position[-1,1],marker='o',color='black')

	# plt.plot([start[0], p.position[1,0]],[start[1],p.position[1,1]],color='green',lw=3)
	for i in range(2,len(p.position),1):
		a = p.position[i-2]
		b = p.position[i-1]
		c = p.position[i]
		x1,y1,x2,y2 = (c[0]+b[0])/2,(c[1]+b[1])/2,(b[0]+a[0])/2,(b[1]+a[1])/2
		xs,ys = [x1,x2],[y1,y2]
		plt.plot(xs,ys,color='b',lw=3)
	plt.plot([p.position[-1,0],goal[0]],[p.position[-1,1],goal[1]],color='black',lw=3)
	plt.plot(p.position[-1,0],p.position[-1,1],marker='o',color='yellow')

	startCircle = plt.Circle((start[0],start[1]),start[2],color='green')
	ax1.add_artist(startCircle)

	goalCircle = plt.Circle((goal[0],goal[1]),goal[2],color='black')
	ax1.add_artist(goalCircle)

	plt.show()
