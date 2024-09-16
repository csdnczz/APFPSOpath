import numpy as np


x_lim = (-30,30)
y_lim = (-30,30)

low = -100
high = 100

init_low_x = -20
init_high_x = -19
init_low_y = -20
init_high_y = -19


# print(rand1)
n_particles = 8

n_iterations = 500
interval = 1

goal = (20,24,3)
start = (-28,-28,2)
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

# obstacles = [(i,j,2) for i in range(-25,25,8) for j in range(-25,25,8)]

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


best = (0.96, 0.01, 0.08)
demo = (0.95, 0.001, 0.1)

def computeField(particle):
	# print(particle.position)
	p_pos = particle.position[-1]
	x = p_pos[0]
	y = p_pos[1]
	energy = (np.square(x-goal[0]) + np.square(y-goal[1]))*200
	for o in obstacles:
		energy -= (np.square(x-o[0]) + np.square(y-o[1]))*2
	return energy


def update_velocity(particle, wM, wL, wG):
	global g_value, g_position
	
	new_velocity =	wM*particle.velocity + \
					wL*(particle.b_position - particle.position[-1]) + \
					wG*(g_position - particle.position[-1])
	
	for o in obs_np:
		distance = np.linalg.norm(particle.position[-1] - np.array([o[:2]]))
		new_velocity += (particle.position[-1] - np.array(o[:2]))/(12*np.square(distance - o[2]))
		new_velocity += np.random.normal(size=(2,))/35


	return new_velocity/np.linalg.norm(new_velocity)

def update_position(particle):
	p_vel = particle.velocity
	new_position = particle.position[-1] + p_vel

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

