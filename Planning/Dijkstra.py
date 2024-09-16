from heapq import heapify, heappush, heappop

# from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.pylab import *
import os
import datetime

plt.style.use('seaborn-pastel')

# Sent for figure
font = {'size': 12}
matplotlib.rc('font', **font)

x_lim = (0, 120)
y_lim = (0, 120)

n_iterations = 20000
interval = 1

openSet = []
heapify(openSet)

fig = plt.figure(figsize=(12, 12))
ax1 = subplot2grid((1, 1), (0, 0))
extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

t_now = datetime.datetime.now()
tstamp = str(t_now.date()) + '_' + str(t_now.hour) + '_' + str(t_now.minute) + '_' + str(t_now.second)
os.mkdir('./Images/Dijkstra_' + tstamp)
os.mkdir('./Residuals/Dijkstra_' + tstamp)

goal = (100, 108, 6)
start = np.array([4, 4])
current = np.array([start[0], start[1]])
obstacles = [(-15, 10, 5),
             (-3, 16, 5),
             (17, 5, 2),
             (1, -3, 4),
             (0, -20, 6),
             (20, 12, 4),
             (-14, -18, 4),
             (13, -12, 4),
             (-10, -8, 4),
             (14, -20, 2),
             (8, 5, 4),
             (-22, -3, 4),
             (10, 20, 3),
             (23, -2, 4)
             ]

for i in range(len(obstacles)):
    obstacles[i] = (obstacles[i][0] * 2 + 60, obstacles[i][1] * 2 + 60, obstacles[i][2] * 2)

# obstacles = [(i,j,2) for i in range(-25,25,8) for j in range(-25,25,8)]

obs_np = list(map(np.array, obstacles))

goal_reached = False

a = []
b = []
pa, pb = [], []

visited = np.array([[0 for i in range(120)] for j in range(120)])
error = [10000000]


ds = [2 * np.array([1, 1]),
      2 * np.array([1, 0]),
      2 * np.array([1, -1]),
      2 * np.array([0, -1]),
      2 * np.array([-1, -1]),
      2 * np.array([-1, 0]),
      2 * np.array([-1, 1]),
      2 * np.array([0, 1])]


def goal_check(pos):
    distance = np.linalg.norm(pos - np.array([goal[:2]]))
    print(pos, goal[2], distance)
    if distance < goal[2]:
        return True
    else:
        return False


def free(pos):
    good = True
    for o in obstacles:
        distance = np.linalg.norm(pos - np.array([o[:2]]))
        if distance < o[2]:
            good = False
            break
    return good


count = 0

def compute_residual(pos):
    """
    :param particle:
    :return:
    """
    x = pos[0]
    y = pos[1]
    energy = (np.square(x - goal[0]) + np.square(y - goal[1])) * 200
    for o in obstacles:
        distance = np.linalg.norm(pos - o[:2])
        energy -= np.square(distance) * 2
    return energy

def update(i):
    global current, openSet, dist, prev, count, goal_reached, error
    print("Update:", i)

    res = compute_residual(current)
    if res < error[-1]:
        error.append(res)
    else:
        error.append(error[-1])

    if len(openSet) > 0 and not (goal_reached):
        current = heappop(openSet)[2]
        a.append(current[0])
        b.append(current[1])
    if goal_check(current) or goal_reached:
        print("Goal Reached!")
        goal_reached = True

        total_path.append(current)

        pa.append(current[0])
        pb.append(current[1])

        path.set_data(pa, pb)
        fig.savefig('Images/Dijkstra_' + tstamp + '/Dijkstra.png', bbox_inches=extent.expanded(1.2, 1.2))
        np.savetxt('Residuals/Dijkstra_' + tstamp + '/Dijkstra_Residual.csv', np.asarray(error) / 1000, delimiter=', ')

        print((start == current).all(), start, current)
        if (start != current).all():
            current = prev[current[0], current[1]]
        else:
            return graph, path
            sys.exit()

    else:
        for d in ds:
            count += 1
            nextNode = current + d
            if (x_lim[0] < nextNode[0] < x_lim[1]) and (y_lim[0] < nextNode[1] < y_lim[1]) and visited[
                nextNode[0], nextNode[1]] == 0 and free(nextNode):

                alt = dist[current[0], current[1]] + np.linalg.norm(ds)

                if alt < dist[nextNode[0], nextNode[1]]:
                    print("Reached")
                    dist[nextNode[0], nextNode[1]] = alt
                    prev[nextNode[0], nextNode[1]] = current

                    if nextNode not in openSet:
                        heappush(openSet, (alt, count, nextNode))

        graph.set_data(a, b)

    return graph, path


if __name__ == "__main__":
    global dist, prev

    dist = np.ones(shape=(120, 120)) * math.inf
    prev = np.array([[start for i in range(120)] for j in range(120)])
    total_path = []

    dist[start[0], start[1]] = 0

    graph, = ax1.plot([], [], 'o', markersize='4.5', color='blue')
    path, = ax1.plot([], [], 'o', markersize='6.0', color='yellow')

    margin = 5.0
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 120)
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")

    ax1.grid(False)

    for o in obstacles:
        circle = plt.Circle((o[0], o[1]), o[2], color='r')
        ax1.add_artist(circle)

    goalCircle = plt.Circle((goal[0], goal[1]), goal[2], color='black')
    ax1.add_artist(goalCircle)

    startCircle = plt.Circle((start[0], start[1]), 2, color='green')
    ax1.add_artist(startCircle)

    x = np.arange(0, 120, 1)
    y = np.arange(0, 120, 1)

    X, Y = np.meshgrid(x, y)

    scat = plt.scatter(X, Y, s=0.6, color='green')

    simulation = FuncAnimation(fig, update, blit=True, frames=n_iterations, interval=interval, repeat=False)
    plt.show()
