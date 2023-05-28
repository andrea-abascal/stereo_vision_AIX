import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def get_pts(plyfile):
	data = np.loadtxt(plyfile,'str' ,delimiter= None)
    
	return data[12:,0], data[12:,1], data[12:,2] #returns X,Y,Z points skipping the first 12 lines
	
def plot_ply(plyfile):
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x,y,z = get_pts(plyfile)
	ax.scatter(x, y, z, c='r', marker='o')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()	

plyfile = get_pts('results/pointcloud.ply')


plot_ply(plyfile)