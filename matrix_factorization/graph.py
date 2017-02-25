from matplotlib import pyplot
from pylab import genfromtxt  
mat0 = genfromtxt("vary_alpha")
 #mat1 = genfromtxt("data1.txt");
pyplot.plot(mat0[:,0], mat0[:,1], label = "data0")
 #pyplot.plot(mat1[:,0], mat1[:,1], label = "data1");
pyplot.legend()
pyplot.show()
