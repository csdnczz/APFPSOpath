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
font = {'size' : 18}
matplotlib.rc('font', **font)
x_lim = (0,100)
y_lim = (-30,30)
low = -100
high = 100
init_low_x = -50
init_high_x = -49
init_low_y = -50
init_high_y = -49
# print(rand1)
n_particles = 12
n_iterations = 300
interval = 30
pdots_x = [0 for i in range(n_particles)]
pdots_y = [0 for i in range(n_particles)]
lines = []
guides = []
particles = []
g_value = 1000000
g_position = np.array([low + 2*high*np.random.random(), low + 2*high*np.random.random()])
error = []
def rastrigin(pos):
# paraboloid = pos[0]**2 + pos[1]**2
x = pos[0]+10
y = pos[1]+20
rastrigin = 20 + (x)**2 + (y)**2 - 10*np.cos(2*np.pi*(x)) - 10*np.cos(2*np.pi*y)
return rastrigin
def update_velocity(particle, w0, w1, w2):
global g_value, g_position
new_velocity = w0*particle.velocity + \
w1*(particle.b_position - particle.position[-1]) + \
w2*(g_position - particle.position[-1])
return new_velocity
class Particle:
def __init__(self, id):
self.id = id
self.position = np.array([[init_low_x+ 2*init_high_x*np.random.random(),
init_low_y + 2*init_high_y*np.random.random()]])
self.velocity = np.array([0.0, 0.0])
self.b_value = math.inf
self.b_position = self.position[-1]
def __str__(self):
return "Particle {}: Position:{} Velocity:{} bVal:{} bPos{}".format(self.id,
self.position,
self.velocity,
self.b_value,
self.b_position)
def update(i):
global data, g_value, g_position, error, line2, lines, guides, pdots
print(i, g_position, g_value)
for k in range(n_particles):
# Calculate target value
f = rastrigin(particles[k].position[-1])
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
if k%3 == 0:
diff = particles[k+1].position[-1] - particles[k+2].position[-1]
new_particle_position = 0.5*particles[k+1].position[-1] +
0.5*particles[k+2].position[-1] + (np.array([diff[1],-diff[0]]) - 2.0 +
4.0*np.array([np.random.random(),np.random.random()]))
else:
new_particle_position = particles[k].position[-1] + particles[k].velocity
# append new position
particles[k].position = np.vstack([particles[k].position,
new_particle_position])
a = particles[k].position[:,0]
b = particles[k].position[:,1]
# print(error)
lines[k].set_data(a,b)
# pdots_x[k] = a[-1]
# pdots_y[k] = b[-1]
# pdots.set_data(np.asarray(pdots_x), np.asarray(pdots_y))
if k%3 == 0:
guides[3+int(k/3)].set_data([particles[k].position[-1,0],
particles[k+1].position[-1,0]],[particles[k].position[-1,1],particles[k+1].position[-1,1]])
guides[int(k/3)].set_data([particles[k+1].position[-1,0],
particles[k+2].position[-1,0]],[particles[k+1].position[-1,1],particles[k+2].position[-1,1]])
# lines[-1].set_data(time, error)
time = np.linspace(1,i+2,i+2)
error.append(g_value)
# print(time, np.asarray(error))
line2.set_data(time, np.asarray(error))
# if i+1 == n_iterations:
# print("Iterations:", i+1, "Best Pos:", g_position, "Value:", g_value)
# sys.exit()
return lines[0], line2, tuple(guides), pdots
if __name__ == "__main__":
best = (0.9, 0.11, 0.8)
demo = (0.95, 0.001, 0.05)
w0, w1, w2 = demo
m = 9
fig = plt.figure(figsize=(35,15))
f = lambda x, y: 20 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
xmin, xmax, xstep = low, high, 0.05
ymin, ymax, ystep = low, high, 0.05
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep,
ystep))
z = f(x+10, y+20)
# Plot 1: Contour and PSO particle animation
ax1 = subplot2grid((1,2),(0,0))
# ax1 = plt.axes(xlim=(low, high), ylim=(low, high))
ax1.contour(x, y, z, levels=np.linspace(0, 200, 30), norm=LogNorm(), cmap=plt.cm.jet)
pdots, = ax1.plot([],[],'ro', markersize=10)
# ax1.grid(True)
# ax1.contour(x, y, z, levels=np.linspace(0, 200, 30), norm=LogNorm(), cmap=plt.cm.jet)
for i in range(n_particles):
line_i, = ax1.plot([], [], lw=2)
lines.append(line_i)
for i in range(int(n_particles/3)):
line_i, = ax1.plot([], [], lw=6, color='green')
guides.append(line_i)
for i in range(3+int(n_particles/3)):
line_i, = ax1.plot([], [], lw=6, color='yellow')
guides.append(line_i)
# Plot 2: Overall loss function plot vs time
ax2 = subplot2grid((1,2),(0,1))
line2, = ax2.plot([],[],lw=3)
# lines.append(line2)
# print(len(lines))
margin = 5.0
ax1.set_xlim(-10-margin,-10+margin)
ax1.set_ylim(-20-margin,-20+margin)
ax1.set_xlabel("M-axis (Line Slope)")
ax1.set_ylabel("C-axis (Line Y-Intercept)")
ax2.set_xlim(0,200)
ax2.set_ylim(0,200)
ax2.set_xlabel("Iterations Elapsed (PSO)")
ax2.set_ylabel("Total Residual (PSO)")
ax1.grid(True)
ax2.grid(True)
for k in range(n_particles):
particles.append(Particle(k))
simulation = FuncAnimation(fig, update, blit=False, frames=n_iterations,
interval=interval, repeat=False)
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
plt.show()
