# from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.pylab import *
import os
import datetime

plt.style.use('seaborn-pastel')

# Sent for figure
font = {'size': 12}
matplotlib.rc('font', **font)

x_lim = (-30, 30)
y_lim = (-30, 30)

low, high = -100, 100

init_low_x, init_high_x, init_low_y, init_high_y = -20, -19, -20, -19
n_particles, n_iterations, interval = 30, 500, 1

goal_reached = False
goal = (20, 24, 3)
start = (-28, -28, 2)
obstacles = [(-15, 10, 5), (-3, 16, 5), (17, 5, 2), (1, -3, 4), (0, -20, 6),
             (20, 12, 4), (-14, -18, 4), (13, -12, 4), (-10, -8, 4), (14, -20, 2),
             (8, 5, 4), (-22, -3, 4), (10, 20, 3), (23, -2, 4)]

line_obstacles = [(-30, 0, 15, 0), (20, -10, 30, -10), (15, 0, 15, 5), (20, -10, 20, 10),
                  (20, 10, 5, 10)]

# obstacles = [(i,j,3) for i in range(-25,25,16) for j in range(-25,25,16)]
obs_np = list(map(np.array, obstacles))

pdots_x = [0 for i in range(n_particles)]
pdots_y = [0 for i in range(n_particles)]
lines, guides, particles = [], [], []

g_value = 1000000
g_position = np.array([low + 2 * high * np.random.random(), low + 2 * high * np.random.random()])
error = []

best = (0.96, 0.01, 0.08)
demo = (0.95, 0.01, 0.1)

wM, wL, wG = demo
m = 9

fig = plt.figure(figsize=(24, 12))
ax1 = subplot2grid((1, 2), (0, 0))
ax2 = subplot2grid((1, 2), (0, 1))
line2, = ax2.plot([], [], lw=1)

extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

pathParticleID, pathFindTime = 0, 0
vStart, vEnd = [], []
startPoint = np.array(start[:2])

segCount = 0
segments = []

t_now = datetime.datetime.now()
tstamp = str(t_now.date()) + '_' + str(t_now.hour) + '_' + str(t_now.minute) + '_' + str(t_now.second)
os.mkdir('./Images/' + tstamp)
os.mkdir('./Residuals/' + tstamp)


class Particle:
    def __init__(self, id):
        self.id = id
        self.position = [np.array(
            [init_low_x + 2 * init_high_x * np.random.random(), init_low_y + 2 * init_high_y * np.random.random()])]
        self.velocity = np.array([0.0, 0.0])
        self.b_value = math.inf
        self.b_position = self.position[-1]

    def __str__(self):
        return "Particle {}: Position:{} Velocity:{} bVal:{} bPos{}".format(self.id, self.position, self.velocity,
                                                                            self.b_value, self.b_position)


def scan_lidar(particle):
    """
    Calculate and return a LIDAR-type scan around the given particle.
    Consider all obstacle types such as linear and circular.
    :param particle:
    :return:
    """
    for i in range(60):
        print("None")


def compute_field(particle):
    """
    Compute the PSO fitness function at the particle position.
    Combine both distances from goal and obstacles.
    Higher is better.
    :param particle:
    :return:
    """

    p_pos = particle.position[-1]
    x = p_pos[0]
    y = p_pos[1]
    energy = (np.square(x - goal[0]) + np.square(y - goal[1])) * 200
    for o in obstacles:
        distance = np.linalg.norm(p_pos - o[:2])
        energy -= np.square(distance) * 2
    return energy


def point_projection(p, a, b):
    """
    Calculate the orthogonal projection of point 'p' on line 'a-b'.
    :param p:
    :param a:
    :param b:
    :return:
    """
    l = np.dot(p - a, b - a)
    ab = np.linalg.norm(b - a)
    ab2 = np.square(ab)
    m = a + l * (b - a) / ab2
    return m


v_obs = (0, 0, 0)


def apply_temporal_field(particle, new_velocity):
    """
    Create a virtual circular obstacle at the 5th last position of the given particle.
    Use this as a temporal obstacle which is applied only when the distance between
    0th and h'th previous positions of the particle is larger than a threshold.
    :param particle:
    :return:
    """
    global v_obs

    h = 20
    if len(particle.position) < h:
        return new_velocity

    travel = np.linalg.norm(particle.position[-h] - particle.position[-1])

    if travel < 1:
        o = (particle.position[-int(h / 2)][0], particle.position[-int(h / 2)][1], 1)
        d_last = np.linalg.norm(np.array(o[:2]) - np.array(v_obs[:2]))

        if d_last > 5:
            print("Temporal Field Applied")
            v_obs = o

        distance = np.linalg.norm(particle.position[-1] - np.array([v_obs[:2]]))
        new_velocity += 30 * (particle.position[-1] - np.array(v_obs[:2])) / (np.square(distance - v_obs[2]))

    return new_velocity


def apply_circ_field(particle, new_velocity):
    """
    Compute velocity change due to potential fields from circular obstacles.
    Lower is better.
    :param particle:
    :param new_velocity:
    :return:
    """
    dmargin = 0.4
    for o in obs_np:
        distance = np.linalg.norm(particle.position[-1] - np.array([o[:2]])) - dmargin
        new_velocity += (particle.position[-1] - np.array(o[:2])) / (12 * np.square(distance - o[2]))

    return new_velocity


def apply_line_field(particle, new_velocity):
    """
    Compute change in particle's velocity due to linear obstacles such as walls and boxes.
    Line obstacles exert acceleration perpendicular to their slope going away from them.
    :param particle:
    :param new_velocity:
    :return:
    """
    for o in line_obstacles:
        p, a, b = particle.position[-1], np.array(o[:2]), np.array(o[2:])
        m = point_projection(p, a, b)

        od = np.linalg.norm(p - m)
        al, bl = np.linalg.norm(m - a), np.linalg.norm(m - b)
        if od < 2 and np.abs(al - bl) < np.linalg.norm(b - a):
            new_velocity += (p - m) / (1e-16 + np.square(0.5 * (od)))

        # ax1.plot([m[0],p[0]],[m[1],p[1]],'r-')

    return new_velocity


def update_velocity(particle, wM, wL, wG):
    """
    Compute the final velocity update for the given particle.
    Combine velocity updates from linear, circular and other obstacles.
    :param particle:
    :param wM:
    :param wL:
    :param wG:
    :return:
    """
    global g_value, g_position
    new_velocity = wM * particle.velocity + \
                   wL * (particle.b_position - particle.position[-1]) + \
                   wG * (g_position - particle.position[-1])
    new_velocity = apply_circ_field(particle, new_velocity)
    # new_velocity = apply_temporal_field(particle, new_velocity)
    new_velocity += np.random.normal(size=(2,)) / 10
    return new_velocity / np.linalg.norm(new_velocity)


def update_position(particle):
    """
    Compute final position update for the given particle.
    Combine effects from only the velocity update.
    Consider reflecting the particle velocity at boundaries.
    :param particle:
    :return:
    """
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
    """
    Check whether the distance to goal is within the goal threshold.
    :param particle:
    :return:
    """
    distance = np.linalg.norm(particle.position[-1] - np.array([goal[:2]]))
    if distance < goal[2]:
        return True
    else:
        return False


def visible(start, end):
    """
    Check whether the 'end' point is visible from the 'start' point.
    Check collision with all circular obstacles.
    :param start:
    :param end:
    :return:
    """
    result = True
    margin = 0.3

    for o in obstacles:
        obs = o[:2]
        l2 = np.square(np.linalg.norm(end - start))
        t = max(0, min(1, dot(obs - start, end - start) / l2))
        projection = start + t * (end - start)
        d = np.linalg.norm(obs - projection)
        # print(d,o[2])
        if d < o[2] + margin:
            result = False
            break

    return result


def visible_line(start, end):
    """
    Check whether the 'end' point is visible from the 'start' point.
    Check collision with all linear obstacles such as walls and boxes.
    :param start:
    :param end:
    :return:
    """
    result = True
    margin = 0.3

    for o in line_obstacles:
        a, b = np.array(o[:2]), np.array(o[2:])
        sm = point_projection(start, a, b)
        em = point_projection(end, a, b)
        smd = start - sm
        emd = end - em

        am = point_projection(a, start, end)
        bm = point_projection(b, start, end)
        amd = a - am
        bmd = b - bm

        prod_ab = np.dot(amd, bmd)
        prod_se = np.dot(smd, emd)

        if prod_ab < 2 and prod_se < 2:
            result = False
            break

    return result


def count_files(name):
    files = os.listdir('./Images')
    a = 0
    for f in files:
        print(f)
        if f[:len(name)] == name:
            a += 1

    return a


def update(i):
    """
    Event loop for the algorithm. All the updates are called from this function
    which runs 'n_iteration' number of times.
    :param i:
    :return:
    """
    global g_value, g_position, error, line2, lines, guides, goal_reached, \
        pathParticleID, pathFindTime, vStart, vEnd, startPoint, pathLines, \
        segCount

    if not (goal_reached):
        print("Update:{}".format(i))
        for k in range(n_particles):

            f = compute_field(particles[k])
            if f < particles[k].b_value:
                particles[k].b_value = f
                particles[k].b_position = particles[k].position[-1]

            if f < g_value:
                g_value = f
                g_position = particles[k].position[-1]

            particles[k].velocity = update_velocity(particles[k], wM, wL, wG)
            new_particle_position = update_position(particles[k])
            particles[k].position.append(new_particle_position)

            if goal_check(particles[k]):

                for p in particles:
                    print(p.position[-1][0])
                    ax1.plot(p.position[-1][0], p.position[-1][1], 'o', color='black', markersize=6)
                ax1.plot(particles[k].position[-1][0], particles[k].position[-1][1], 'o', color='yellow', markersize=6)

                if not (goal_reached):
                    fig.savefig('Images/' + tstamp + '/APF_PSO.png', bbox_inches=extent.expanded(1.2, 1.2))
                goal_reached = True
                pathParticleID = k
                pathFindTime = i + 1

            a = np.array(particles[k].position)[:, 0]
            b = np.array(particles[k].position)[:, 1]
            lines[k].set_data(a, b)

        time = np.linspace(1, i + 2, i + 2)
        error.append(g_value)
        line2.set_data(time, np.asarray(error) / 1000)
        return lines[0], line2, tuple(guides)

    else:
        path = particles[pathParticleID].position

        t = i - pathFindTime

        if t < len(path):
            currentPoint = path[t]
            vStart.append(startPoint)
            vEnd.append(currentPoint)
            vS = np.array(vStart)
            vE = np.array(vEnd)

            ax1.plot([vS[-1, 0], vE[-1, 0]], [vS[-1, 1], vE[-1, 1]], 'r-', lw=0.5)

            if not (visible(startPoint, currentPoint)):
                ax1.plot([vS[-1, 0], vE[-1, 0]], [vS[-1, 1], vE[-1, 1]], 'g-', lw=2)
                segments.append([startPoint, currentPoint])
                startPoint = currentPoint
                segCount += 1

            if t == 80:
                fig.savefig('Images/' + tstamp + '/Visible.png', bbox_inches=extent.expanded(1.2, 1.2))

        if t == len(path):
            print("Total Segments:", len(segments))
            segments.append([startPoint, goal[:2]])
            for s, e in segments:
                ax1.plot([s[0], e[0]], [s[1], e[1]], 'b-', lw=3)

            fig.savefig('Images/' + tstamp + '/FinalPath.png', bbox_inches=extent.expanded(1.2, 1.2))
            np.savetxt('Residuals/' + tstamp + '/APF_PSO_Residual.csv', np.asarray(error)/1000, delimiter=', ')

        return lines[0], line2, tuple(guides)


def draw_circ_obs():
    for o in obstacles:
        circle = plt.Circle((o[0], o[1]), o[2], color='red')
        ax1.add_artist(circle)


def draw_line_obs():
    for o in line_obstacles:
        ax1.plot([o[0], o[2]], [o[1], o[3]], 'g-', lw=2)


if __name__ == "__main__":
    # global fig

    for i in range(n_particles):
        line_i, = ax1.plot([], [], lw=2)
        lines.append(line_i)

    margin = 5.0
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-30, 30)
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")

    ax2.set_xlim(0, 500)
    ax2.set_ylim(-100, 200)
    ax2.set_xlabel("Iterations Elapsed (PSO)")
    ax2.set_ylabel("Total Residual (PSO)")
    ax1.grid(False)
    ax2.grid(True)

    x = np.arange(-30, 30, 2)
    y = np.arange(-30, 30, 2)
    X, Y = np.meshgrid(x, y)
    u = ((X - goal[0]) ** 2) * 10 + 1
    v = ((Y - goal[1]) ** 2) * 10 + 1

    ax1.quiver(X, Y, u, v, color='green', lw=1)

    # draw_line_obs()
    draw_circ_obs()

    goalCircle = plt.Circle((goal[0], goal[1]), goal[2], color='black')
    ax1.add_artist(goalCircle)
    startCircle = plt.Circle((start[0], start[1]), start[2], color='green')
    ax1.add_artist(startCircle)

    for k in range(n_particles):
        particles.append(Particle(k))

    anim = FuncAnimation(fig, update, blit=False, frames=n_iterations, interval=interval, repeat=False)
    plt.show()

    # print(count_files("APF_PSO_Visible"))
