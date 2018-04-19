import sys
import time
#import numpy as np

def main():
	(training_data,testing_data) = parse_data() if len(sys.argv) != 3 else parse_data(sys.argv[1],sys.argv[2])
	normalize(training_data,testing_data)

def parse_data(training_filename = "knn_train.csv", testing_filename = "knn_test.csv"):
	training_file = open(training_filename, 'r')
	test_file = open(testing_filename, 'r')
	
	start_time0 = time.clock()
	training_data = [[float(item) for item in line.split(',')] for line in training_file.read().splitlines()]
	testing_data = [[float(item) for item in line.split(',')] for line in test_file.read().splitlines()]
	training_file.close()
	test_file.close()
	return (training_data, testing_data)

def normalize(training_data,testing_data):
   	mins = training_data[0][1:]
	maxs = training_data[0][1:]
	for point in training_data[1:]:
	   new_mins = [min(mins[i],point[i+1]) for i in xrange(0,len(mins))]
	   new_maxs = [max(maxs[i],point[i+1]) for i in xrange(0,len(maxs))]
	   mins = new_mins
	   maxs = new_maxs
	for point in training_data:
	   for i in xrange(1,len(point)):
	      point[i] = (point[i]-mins[i-1])/(maxs[i-1]-mins[i-1])
	for point in testing_data:
	   for i in xrange(1,len(point)):
	      point[i] = (point[i]-mins[i-1])/(maxs[i-1]-mins[i-1])
	
if __name__ == "__main__": main()
