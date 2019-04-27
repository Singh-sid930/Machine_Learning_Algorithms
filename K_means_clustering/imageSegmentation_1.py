'''
    IMAGE SEGMENTATION USING K-MEANS (UNSUPERVISED LEARNING)
    AUTHOR Paul Asselin

    command line arguments:
		python imageSegmentation.py K inputImageFilename outputImageFilename
	where K is greater than 2
'''

import numpy as np
import sys
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

iterations = 5

#	Parse command-line arguments
#	sets K, inputName & outputName

K = int(sys.argv[1])
if K < 3:
	print ("Error: K has to be greater than 2")
	sys.exit()
input_image = sys.argv[2]
output_image = sys.argv[3]

#	Open input image
image = Image.open(input_image)
image_width = image.size[0]
image_height = image.size[1]

#	Initialise data vector with attribute r,g,b,x,y for each pixel
pixel_matrix = np.ndarray(shape=(image_width * image_height, 5), dtype=float)


#	Initialise vector that holds which cluster a pixel is currently in
pixel_cluster_vector = np.ndarray(shape=(image_width * image_height), dtype=int)

#	Populate data vector with data from input image
for y in range(0, image_height):
      for x in range(0, image_width):
      	coord = (x, y)
      	rgb = image.getpixel(coord)
      	pixel_matrix[x + y * image_width, 0] = rgb[0]
      	pixel_matrix[x + y * image_width, 1] = rgb[1]
      	pixel_matrix[x + y * image_width, 2] = rgb[2]
      	pixel_matrix[x + y * image_width, 3] = x
      	pixel_matrix[x + y * image_width, 4] = y

#	Standarize the values of features
pixel_matrix_scaled = preprocessing.normalize(pixel_matrix)

#	Set centers
minValue = np.amin(pixel_matrix_scaled)
maxValue = np.amax(pixel_matrix_scaled)

centers = np.ndarray(shape=(K,5))
for index, center in enumerate(centers):
	centers[index] = np.random.uniform(minValue, maxValue, 5)

for iteration in range(iterations):
	#	Set pixels to their cluster
	for idx, data in enumerate(pixel_matrix_scaled):
		distanceToCenters = np.ndarray(shape=(K))
		for index, center in enumerate(centers):
			distanceToCenters[index] = euclidean_distances(data.reshape(1, -1), center.reshape(1, -1))
		pixel_cluster_vector[idx] = np.argmin(distanceToCenters)

	##################################################################################################
	#	Check if a cluster is ever empty, if so append a random datapoint to it
	clusterToCheck = np.arange(K)		#contains an array with all clusters
										#e.g for K=10, array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	clustersEmpty = np.in1d(clusterToCheck, pixel_cluster_vector)
										#^ [True True False True * n of clusters] False means empty
	for index, item in enumerate(clustersEmpty):
		if item == False:
			pixel_cluster_vector[np.random.randint(len(pixel_cluster_vector))] = index
			# ^ sets a random pixel to that cluster as mentioned in the homework writeup
	##################################################################################################

	#	Move centers to the centroid of their cluster
	for i in range(K):
		dataInCenter = []

		for index, item in enumerate(pixel_cluster_vector):
			if item == i:
				dataInCenter.append(pixel_matrix_scaled[index])
		dataInCenter = np.array(dataInCenter)
		centers[i] = np.mean(dataInCenter, axis=0)




#	set the pixels on original image to be that of the pixel's cluster's centroid
for index, item in enumerate(pixel_cluster_vector):
	pixel_matrix[index][0] = int(round(centers[item][0] * 255))
	pixel_matrix[index][1] = int(round(centers[item][1] * 255))
	pixel_matrix[index][2] = int(round(centers[item][2] * 255))

#	Save image
image = Image.new("RGB", (image_width, image_height))

for y in range(image_height):
	for x in range(image_width):
	 	image.putpixel((x, y), (int(pixel_matrix[y * image_width + x][0]), 
	 							int(pixel_matrix[y * image_width + x][1]),
	 							int(pixel_matrix[y * image_width + x][2])))
image.save(output_image+".jpeg")