import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import toimage
import random

def load_data(filename):
   f = open(filename, 'r')
   lines = f.read().splitlines()
   f.close()
   return np.array([[float(item) for item in line.split(',')] for line in lines])

def main():
   data = load_data('unsupervised.txt')
   (partitions, centers, SSEs) = kmeans(data, k=2)

#performs the kmeans algorithm
#returns the partitions as a list of indices for each item in data, the center of mass of each partition, and the SSE at each iteration
def kmeans(data, k=2):
   length = len(data)
   #choose k random points for seeding
   centers = [data[ind] for ind in np.random.choice(length,k)]
   done = False
   SSEs = []
   partitions = [-1 for i in xrange(0,length)] #mark each partition as unset
   while not done:
      done = True
      new_partitions = [-1 for i in xrange(0,length)]
      for item in xrange(0,length):
	 best_partition = 0
	 best_distance = np.linalg.norm(data[item]-centers[0])
	 for i in xrange(1,k):
	    distance = np.linalg.norm(data[item]-centers[i])
	    if distance < best_distance:
	       best_distance = distance
	       best_partition = i
	 #mark which partition it belongs to
	 new_partitions[item] = best_partition
	 if new_partitions[item] != partitions[item]: 
	    #if there is a change, we have not converged yet
	    done = False
      partitions = new_partitions
      #calculate new centers of mass
      counts = [0 for i in xrange(0,k)]
      temp_centers = [np.zeros(len(data[0])) for i in xrange(0,k)]
      for item in xrange(0,length):
	 counts[partitions[item]] = counts[partitions[item]] + 1
	 temp_centers[partitions[item]] = temp_centers[partitions[item]] + data[item]
      for i in xrange(0,k):
	 temp_centers[i] = temp_centers[i]/counts[i]
      centers = temp_centers
      SSEs.append(calculate_SSE_kmeans(data,centers,partitions))
   return partitions, centers, SSEs

def calculate_SSE_kmeans(data,centers,partitions):
   SSE = 0.0
   for i in xrange(0, len(data)):
      SSE = SSE + math.pow(np.linalg.norm(data[i]-centers[partitions[i]]),2)
   return SSE
main()
