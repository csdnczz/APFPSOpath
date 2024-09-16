import numpy as np
import matplotlib.pyplot as plt




fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# w = np.array([5, -7, 5, 5])
#
# xx, yy = np.meshgrid(np.arange(-10,15,2), np.arange(-10,40,2))
# gt_plane = (-w[0] * xx - w[1] * yy - w[3]) * (1. / w[2])
# pc = gt_plane + np.random.normal(0.0, 4, size=gt_plane.shape)
#
# params = np.array([2, -2, 4, 8])
# z = (-params[0] * xx - params[1] * yy - params[3]) * (1. / params[2])
#
#
# f = np.array([xx, yy, pc])
# f = f.swapaxes(0,2)
# print(f.shape)
#
# a = []
# for i in range(f.shape[0]):
#     for j in range(f.shape[1]):
#         a.append(f[i,j,:])
# a = np.array(a)
# print("PC:", a)

b = np.loadtxt('/home/quantum/Workspace/FastStorage/catkin_ws/src/MapSenseROS/scripts/data.txt', delimiter=',', dtype=np.float64)

c = np.mean(b, axis=0)
b -= c

# ones = np.ones(shape=(a.shape[0],1))
# a = np.hstack([a,ones])
# print(a.shape, ones.shape)

U, S, Vt = np.linalg.svd(b)

xx, yy = np.meshgrid(np.arange(-0.2,0.2,0.04), np.arange(-0.2,0.2,0.04))
plane = (-Vt[-1,0] * xx - Vt[-1,1] * yy - 0.1) * (1. / Vt[-1,2])

print(b)
print(Vt)

ax.scatter(b[:,0], b[:,1], b[:,2])
ax.plot_surface(xx, yy, plane, color='yellow')
plt.show()