import sys
import time
import numpy as np
import heapq
import math

def main():
	(training_data,testing_data) = parse_data() if len(sys.argv) != 3 else parse_data(sys.argv[1],sys.argv[2])
	normalize(training_data,testing_data)
	#run_knn(training_data,testing_data)
	for d in xrange(0,10):
		tree = decision_tree(training_data,d)
		#tree.print_tree()
		training_correct_count = 0
		testing_correct_count = 0
		for point in training_data:
		   if tree.get_choice(point) == point[0]:
		      training_correct_count = training_correct_count+1
		for point in testing_data:
		   if tree.get_choice(point) == point[0]:
		      testing_correct_count = testing_correct_count+1
		print "depth =",d,":"
		print training_correct_count,"/",len(training_data),"=",float(training_correct_count)*100.0/float(len(training_data)),"% correct for training data"
		print testing_correct_count,"/",len(testing_data),"=",float(testing_correct_count)*100.0/float(len(testing_data)),"% correct for testing data"
	

class decision_tree():
   	def __init__(self,training_data,depth):
	     	self.leaf = False
		if depth != 0:
		   base_entropy = calculate_entropy(training_data)
		   best_gain = -1
		   best_ind = -1
		   best_thresh = -1
		   for i in xrange(1, len(training_data[0])-1): #i is the index of the feature
		      sorted_set = sorted(training_data, key = lambda point: point[i])
		      for p in xrange(0, len(sorted_set)-1):
			 if sorted_set[p+1][0] != sorted_set[p][0]:
			    split_ind = p
			    while split_ind < len(sorted_set)-1 and sorted_set[split_ind+1][i] == sorted_set[split_ind][i]: # to deal with ties
			       split_ind = split_ind + 1
			    if split_ind < len(sorted_set)-1:
			       left_split = sorted_set[:split_ind+1]
			       right_split = sorted_set[split_ind+1:]
			       info_gain = base_entropy - float(split_ind)/float(len(sorted_set))*calculate_entropy(left_split) - float(len(sorted_set)-split_ind)/float(len(sorted_set))*calculate_entropy(right_split)
			       if info_gain > best_gain:
				  best_gain = info_gain
				  best_ind = i
				  best_thresh = (sorted_set[split_ind][i]+sorted_set[split_ind+1][i])/2.0

		   if best_ind == -1:
		      self.leaf = True
		   else:
		      self.feature_ind = best_ind
		      self.thresh = best_thresh
		      left_list = [point for point in training_data if point[best_ind] <= best_thresh]
		      right_list = [point for point in training_data if point[best_ind] > best_thresh]
		      self.left = decision_tree(left_list,depth-1)
		      self.right = decision_tree(right_list,depth-1)
		else:
		   self.leaf = True
		if self.leaf == True:
		   total = sum([point[0] for point in training_data])
		   if total == 0:
		      self.choice = 0
		   else:
		      self.choice = 1 if total > 0 else -1
	def get_choice(self,point):
	   if self.leaf == True:
	      return self.choice
	   else: #recursively get the choice from the children of this node
	      if point[self.feature_ind] <= self.thresh:
		 return self.left.get_choice(point)
	      else:
		 return self.right.get_choice(point)
	
	def print_tree(self,depth=0):
	   if self.leaf == True:
	      print "-"*depth,self.choice
	   else:
	      print "-"*depth,"point["+str(self.feature_ind)+"] <=",self.thresh
	      self.left.print_tree(depth+1)
	      print "-"*depth,"point["+str(self.feature_ind)+"] >",self.thresh
	      self.right.print_tree(depth+1)

def run_knn(training_data,testing_data):
	for k in [i*2+1 for i in xrange(0,26)]:
	   	print "k =",k
		print "---------------------"
		cv_count = 0
		train_error_count = 0
		for i in xrange(0,len(training_data)):
		   answer = knn(training_data[:i]+training_data[i+1:],training_data[i][1:],k)
		   if answer != int(training_data[i][0]):
		      cv_count = cv_count + 1
		   answer = knn(training_data,training_data[i][1:],k)
		   if (answer != int(training_data[i][0])):
		      train_error_count = train_error_count + 1
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
