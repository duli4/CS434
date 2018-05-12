import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt
#import seaborn as sns

cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

def main():
	#train_sets is a list of 5 dictionaries, each containing two keys: "data" and "labels"
	
	#Data is a 10000x3072 numpy array of uint8s, each row is a 32x32 colour image.
	#The first 1024 entries contain the red channel values, the next 1024 the green,
	#and the final 1024 the blue.
	
	#Labels is a list of 10000 numbers in the range 0-9. The number at index i
	#	indicates the label of the ith image in the array data.
	train_sets = [unpickle("cifar-10-batches-py/data_batch_" + str(i)) for i in xrange(1, 6)]
	
	#Same as train_sets, but a single dictionary with testing data
	test_set = unpickle("cifar-10-batches-py/test_batch")
	
	#The 10 label names associated with the numbers in the "labels" list that each
	#dictionary containing data has.
	meta_set = unpickle("cifar-10-batches-py/batches.meta")
	
	#print [train_sets[i]["data"][0] for i in xrange(5)]
	
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

if __name__ == "__main__":
	main()