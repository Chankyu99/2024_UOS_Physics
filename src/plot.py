import numpy as np
import matplotlib.pyplot as plt
poe=np.genfromtxt('bands.dat.gnu')#,skip_header=1)
plt.scatter(poe[:,0],poe[:,1])
#plt.ylim(-5,5)
plt.show()
