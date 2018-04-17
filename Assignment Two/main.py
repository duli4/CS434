import sys
import time
#import numpy as np

def main():
	parse_data()
	
def parse_data():
	training_file = open("knn_train.csv", 'r')
	test_file = open("knn_test.csv", 'r')
	
	start_time0 = time.clock()
	content = [[float(item) for item in line.split(',')] for line in training_file.read().splitlines()]
	
if __name__ == "__main__": main()