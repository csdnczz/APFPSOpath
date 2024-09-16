import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')


fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
lines = []
n_lines = 4
for i in range(n_lines):
	line_i, = ax.plot([], [], lw=3)
	lines.append(line_i)

def init():
    return tuple(lines)

def animate(i):
    t = np.linspace(0, 4, 1000)
    x = t*np.cos(2*np.pi*(t))
    y = t*np.sin(2*np.pi*(t))
    a = x[:i]
    b = y[:i]


    for k in range(n_lines):
    	lines[k].set_data(a+0.5*k, b)

    return tuple(lines)

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=10, blit=True)

plt.show()