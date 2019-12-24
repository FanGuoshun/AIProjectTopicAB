from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
 
fig = plt.figure()
ax = fig.gca(projection='3d')
 
# Make data.
X = np.arange(-50, 50, 0.1)
Y = np.arange(-50, 50, 0.1)
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X ** 2 + Y ** 2)
# Z = np.sin(R)
A1=X*0.5+Y
B1=1/(1+np.exp(-A1))
A2=X*0.2+Y
B2=1/(1+np.exp(-A2))
A3=X*0.8+Y
B3=1/(1+np.exp(-A3))
A4=X*0.4+Y
B4=1/(1+np.exp(-A4))
A5=X*0.7+Y
B5=1/(1+np.exp(-A5))
Z=((B1-0.1)**2 + (B2-0.8)**2 + (B3-0.6)**2 + (B4-0.1)**2+ (B5-0.9)**2)/10
# Z=((B1-1.2)**2 + (B2-4.2)**2)/4
# Z=((B1-0.34)**2)/2

# Z=-(0.0087*np.log(B1)+(1-0.0087)*np.log(1-B1))/2
 
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
# sur = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)


plt.show()