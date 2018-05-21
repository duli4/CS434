import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import toimage
from collections import OrderedDict
import random

def load_data(filename):
   f = open(filename, 'r')
   lines = f.read().splitlines()
   f.close()
   return np.array([[float(item) for item in line.split(',')] for line in lines])

def main():
   data = load_data('unsupervised.txt')
   #PART 1: K-means with k = 2
   (partitions, centers, SSEs) = kmeans(data, k=2)
   iterations = [i+1 for i in xrange(0, len(SSEs))]
   plt.plot(iterations,SSEs,label = 'k-Means SSE at k=2')
   plt.xlabel('Iterations')
   plt.ylabel('SSE')
   handles, labels = plt.gca().get_legend_handles_labels()
   by_label = OrderedDict(zip(labels, handles))
   plt.legend(by_label.values(), by_label.keys())
   plt.savefig('p1_SSEs')
   plt.close()
   #PART 2: K-means model selection
   best_SSEs = [-1 for k in xrange(2,11)]
   k_choices = [k for k in xrange(2,11)]
   for i in xrange(0, len(k_choices)):
      for attempt in xrange(0,10):
	 (partitions, centers, SSEs) = kmeans(data, k=k_choices[i])
	 if best_SSEs[i] == -1 or SSEs[-1] < best_SSEs[i]:
	    best_SSEs[i] = SSEs[-1]
   plt.plot(k_choices,best_SSEs,label= 'best SSEs for k-Means')
   plt.xlabel('k')
   plt.ylabel('best SSE')
   handles, labels = plt.gca().get_legend_handles_labels()
   by_label = OrderedDict(zip(labels, handles))
   plt.legend(by_label.values(), by_label.keys())
   plt.savefig('p2_SSEs')
   plt.close()


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
	 if counts[i] == 0:
	    temp_centers[i] = centers[i]
	 else:
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
