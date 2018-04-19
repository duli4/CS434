import sys
import time
import numpy as np
import heapq
import math

def main():
	(training_data,testing_data) = parse_data() if len(sys.argv) != 3 else parse_data(sys.argv[1],sys.argv[2])
	normalize(training_data,testing_data)
	#run_knn(training_data,testing_data)
	tree = decision_tree_node()
	tree.split(training_data,1)
	tree.print_tree(0)
	

class decision_tree_node():
   	def __init__(self):
	     	self.leaf = False
	def split(self,data,depth):
	   	if (depth == 0):
		   	self.leaf = True
			self.choice = 1 if sum([data[i][0] for i in xrange(0,len(data))]) > 0 else -1
		else:
			data_entropy = calculate_entropy(data)
			best_gain = 0.0
			best_ind = 1
			best_thresh = 0.5
			for i in xrange(1,len(data[0])):
				sorted_data = sorted(data, key=lambda item:item[i])
				for ind in xrange(1,len(sorted_data)):
				   	if sorted_data[ind][0] != sorted_data[ind-1][0]:
					   gain = data_entropy-float(ind)/float(len(data))*calculate_entropy(data[:ind])-float(len(data)-ind)/float(len(data))*calculate_entropy(data[ind:])
					   if gain > best_gain:
					      best_ind = i
					      best_thresh = (sorted_data[ind-1][i]+sorted_data[ind][i])/2.0
			self.ind = best_ind
			self.thresh = best_thresh
			self.left = decision_tree_node()
			self.right = decision_tree_node()
			sorted_data = sorted(data, key=lambda item:item[self.ind])
			self.left.split(sorted_data[:self.ind],depth-1)
			self.right.split(sorted_data[self.ind:],depth-1)
				
	def get_choice(self,point):
		if self.leaf == True:
		   	return self.choice
		else:
		   	if point[self.ind] < self.thresh:
			   	return self.left.get_choice(point)
			else:
			   	return self.right.get_choice(point)
	def print_tree(self,depth=0):
	   	if (self.leaf == True):
		   	print '-'*depth,"[",self.choice,"]"
		else:
		   	print '-'*depth,"data[",self.ind,"] <",self.thresh
			self.left.print_tree(depth+1)
			print '-'*depth,"data[",self.ind,"] >",self.thresh
			self.right.print_tree(depth+1)

def run_knn(training_data,testing_data):
	for k in [i*2+1 for i in xrange(0,26)]+[75]:
	   	print "k =",k
		print "---------------------"
		cv_count = 0
		train_error_count = 0
		for i in xrange(0,len(training_data)):
			answer = knn(training_data[:i]+training_data[i+1:],training_data[i][1:],k)
			if (answer != int(training_data[i][0])): cv_count = cv_count + 1
			answer = knn(training_data,training_data[i][1:],k)
			if (answer != int(training_data[i][0])): train_error_count = train_error_count + 1
		test_error_count = 0
		for i in xrange(0,len(testing_data)):
		  answer = knn(training_data,testing_data[i][1:],k)
			if (answer != int(testing_data[i][0])): test_error_count = test_error_count + 1
		
		print "training error count:",train_error_count,"/",len(training_data),"=",float(train_error_count*100)/float(len(training_data)),"%"	      
		print "testing error count:",test_error_count,"/",len(testing_data),"=",float(test_error_count*100)/float(len(testing_data)),"%"	      
		print "leave-one-out cross-validation error:",cv_count,"/",len(training_data),"=",float(cv_count*100)/float(len(training_data)),"%"
		

def parse_data(training_filename = "knn_train.csv", testing_filename = "knn_test.csv"):
	training_file = open(training_filename, 'r')
	test_file = open(testing_filename, 'r')
	
	training_data = [[float(item) for item in line.split(',')] for line in training_file.read().splitlines()]
	testing_data = [[float(item) for item in line.split(',')] for line in test_file.read().splitlines()]
	training_file.close()
	test_file.close()
	return (training_data, testing_data)

def normalize(training_data,testing_data):
   	mins = training_data[0][1:]
	maxs = training_data[0][1:]
	for point in training_data[1:]:
	   mins = [min(mins[i],point[i+1]) for i in xrange(0,len(mins))]
	   maxs = [max(maxs[i],point[i+1]) for i in xrange(0,len(maxs))]
	for point in training_data:
	   for i in xrange(1,len(point)):
	      point[i] = (point[i]-mins[i-1])/(maxs[i-1]-mins[i-1])
	for point in testing_data:
	   for i in xrange(1,len(point)):
	      point[i] = (point[i]-mins[i-1])/(maxs[i-1]-mins[i-1])

def knn(training_data,point,k):
   return 1 if sum([points[1] for points in heapq.nsmallest(k,[[np.linalg.norm([data[i+1]-point[i] for i in xrange(0,len(point))]),data[0]] for data in training_data])]) > 0 else -1

def calculate_entropy(data):
   positive_count = 0.0
   negative_count = 0.0
   for item in data:
      if item[0] == 1:
	 positive_count = positive_count + 1.0
      else:
	 negative_count = negative_count + 1.0
   pos = positive_count/(positive_count+negative_count)
   neg = negative_count/(positive_count+negative_count)
   return -(0 if pos == 0 else pos*math.log(pos,2))-(0 if neg == 0 else neg*math.log(neg,2))

if __name__ == "__main__": main()
