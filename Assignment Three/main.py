#Imports
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import OrderedDict

import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt
#import seaborn as sns

#Globals
cuda = torch.cuda.is_available()

class ChunkSampler(data_utils.sampler.Sampler):
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples
	
class Net(nn.Module):
	def __init__(self, dropout_rate = 0.2):
		super(Net, self).__init__()
		# One hidden layer - 100 Nodes
		self.fc1 = nn.Linear(3*32*32, 100)
		self.fc1_drop = nn.Dropout(dropout_rate)
		self.fc2 = nn.Linear(100, 10)
		
		# Two hidden layes - 50 Nodes, 50 Nodes
		# self.fc1 = nn.Linear(3*32*32, 50)
		# self.fc2 = nn.Linear(50, 50)
		# self.fc3 = nn.Linear(50, 10)

	def forward(self, x):
		x = x.view(-1, 3*32*32)
		
		# F.sigmoid can be replaced with F.relu for Part Two
		x = F.sigmoid(self.fc1(x))
		x = self.fc1_drop(x)
		# One hidden layer
		x = self.fc2(x)
		
		# Two hidden layers
		# x = F.sigmoid(self.fc2(x))
		# x = self.fc3(x)
		
		return x

def main():
	print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)	
	
	train_loader, validation_loader, test_loader = get_data()
	for X_train, y_train in train_loader:
		print('X_train:', X_train.size(), 'type:', X_train.type())
		print('y_train:', y_train.size(), 'type:', y_train.type())
		break
	
	# Part One & Two - Change Activation Function in Net Object
	epochs = [i for i in xrange(1)]
	data_vals = 2
	losses = []
	accuracies = []
	test_accuracies = []
	for i in xrange(data_vals):
	   	learn_rate = 0.1*pow(10,-i)
		(net, results) = cte(train_loader, validation_loader, F.nll_loss, learn_rate, 0.9, epochs = 1)
		losses.append([item[0] for item in results])
		accuracies.append([item[1] for item in results])
		test_accuracies.append(test(net,test_loader))
	for i in xrange(data_vals):
	   	print('Model ',i,': ',test_accuracies[i])
	for i in xrange(data_vals):
		plt.plot(epochs, losses[i], label = 'Loss at learning rate ' + str(0.1*pow(10,-i)))
	plt.xlabel('Epoch')
	plt.ylabel('Avg. Loss')
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())
	plt.savefig('p1_losses')
	plt.close()
	for i in xrange(data_vals):
	   	plt.plot(epochs, accuracies[i], label = 'Accuracy at learning rate ' + str(0.1*pow(10,-i)))
	plt.xlabel('Epoch')
	plt.ylabel('Val. Accuracy')
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())
	plt.savefig('p1_accuracies')
	
	# Part Three
	# Play with cte using different drop out, momentum, and weight decay.
	# Try to maximize accuracy / minimize loss
	
	# losses = []
	# accuracies = []
	# test_accuracies = []
	# for i in xrange(6):
		# dropout_rate = .1 + .1*i
		# (net, results) = cte(train_loader, validation_loader, F.nll_loss, 0.1, 0.9, 5, dropout_rate)
		# losses.append([item[0] for item in results])
		# accuracies.append([item[1] for item in results])
		# test_accuracies.append(test(net,test_loader))
	# for i in xrange(6):
		# print('Model ',i,': ',test_accuracies[i])
	# for i in xrange(6):
		# plt.plot(epochs, losses[i], label = 'Loss at dropout_rate ' + str(dropout_rate))
	# plt.xlabel('Epoch')
	# plt.ylabel('Avg. Loss')
	# handles, labels = plt.gca().get_legend_handles_labels()
	# by_label = OrderedDict(zip(labels, handles))
	# plt.legend(by_label.values(), by_label.keys())
	# plt.savefig('losses')
	# plt.close()
	# for i in xrange(6):
		# plt.plot(epochs, accuracies[i], label = 'Accuracy at dropout_rate ' + str(dropout_rate))
	# plt.xlabel('Epoch')
	# ply.ylabel('Val. Accuracy')
	# handles, labels = plt.gca().get_legend_handles_labels()
	# by_label = OrderedDict(zip(labels, handles))
	# plt.legend(by_label.values(), by_label.keys())
	# plt.savefig('accuracies')
	
	
	
	# Part Four
	# Comment out the One hidden layer code, uncomment Two hidden layers
	# Test the same as in parts One & Two
	

def cte(train_loader, validation_loader, loss_function, learn_rate, momentum, epochs = 10, dropout_rate = 0.2):
	net = Net(dropout_rate)
	if cuda:
		net.cuda()
	optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=momentum)
	results = []
	for epoch in range(epochs):
		train(net, train_loader, optimizer, epoch, loss_function)
		results.append(validate(net, validation_loader, loss_function))
	return (net, results)

def test(net, test_loader):
   	net.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))

	return float(correct) / float(total)
	
def train(net, train_loader, optimizer, epoch, loss_function, log_interval=400):
	net.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = net(data)
		loss = loss_function(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader) * train_loader.batch_size,
				100. * batch_idx / len(train_loader), loss.data[0]))
				
def validate(net, validation_loader, loss_function):
	net.eval()
	val_loss, correct = 0, 0
	for data, target in validation_loader:
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = net(data)
		val_loss += loss_function(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	#loss_vector.append(val_loss)

	accuracy = 100. * correct / (len(validation_loader)*validation_loader.batch_size)
	#accuracy_vector.append(accuracy)
	
	print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			val_loss, correct, len(validation_loader)*validation_loader.batch_size, accuracy))

	return val_loss, accuracy

def get_data():
	transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	batch_size = 10
	kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

	NUM_TRAIN = 40000
	NUM_VAL = 10000
	train_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10('./data', train=True, download=True, transform=transform),
		batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN,0), **kwargs)

	validation_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10('./data', train=True, transform=transform),
		batch_size=batch_size, sampler=ChunkSampler(NUM_VAL,NUM_TRAIN), **kwargs)
		
	test_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10('./data', train=False, transform=transform),
		batch_size=batch_size, shuffle=False, **kwargs)
	
	return train_loader, validation_loader, test_loader
	
def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict
	
if __name__ == "__main__":
	main()
