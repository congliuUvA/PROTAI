import numpy as np
import matplotlib.pyplot as plt

soa =np.array([[0,0,0,1,2,3], [0,0,0,1,-2,-3], [0,0,0,1,0,0]])

X, Y, Z, U, V, W = zip(*soa)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
plt.show()