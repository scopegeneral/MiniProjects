import numpy as np
import imageio
import matplotlib.pyplot as plt


def initKMeans(X, K):
	[m, n] = X.shape
	rand_idx = np.random.permutation(m)
	rand_idx = rand_idx[:K]
	centroids = X[rand_idx,:]
	return centroids

def findClosestCentroids(X, initial_centroids):
	[K, n] = initial_centroids.shape
	[m, n] = X.shape
	distances = np.zeros((m,K))
	
	for i in range(K):
		centroid_i = np.array([initial_centroids[i,:],]*m)
		distances_i = (X - centroid_i)**2
		distances[:,i] = distances_i.sum(axis = 1)
	
	idx = np.argmin(distances,axis = 1)
	return idx

def computeCentroids(X, idx, K):
	[m, n] = X.shape
	centroids = np.zeros((K,n))
	for i in range(K):
		group_i = X[idx == i,:]
		centroids[i,:] = group_i.sum(axis = 0)/group_i.shape[0]

	return centroids



if __name__ == "__main__":
	A = imageio.imread(r"C:\Users\320004436\OneDrive - Philips\Desktop\Cp\test3.jpg")
	A = A / 255
	image_size = A.shape
	A = np.reshape(A, (image_size[0]*image_size[1],3))


	K = 16
	max_iter = 10

	centroids = initKMeans(A, K)
	
	for i in range(max_iter):
		print("K-Means Iteration {}/{}".format(i+1,max_iter))
		idx = findClosestCentroids(A, centroids)
		centroids = computeCentroids(A, idx, K)

	
	A_compressed = centroids[idx,:]
	A_compressed = np.reshape(A_compressed,(image_size[0],image_size[1],3))
	plt.figure()
	plt.imshow(A_compressed)
	plt.show()

