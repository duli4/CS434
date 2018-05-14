#Imports
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

#Globals
cuda = torch.cuda.is_available()



class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# One hidden layer - 100 Nodes
		self.fc1 = nn.Linear(3*32*32, 100)
		self.fc2 = nn.Linear(100, 10)
		
		# Two hidden layes - 50 Nodes, 50 Nodes
		# self.fc1 = nn.Linear(3*32*32, 50)
		# self.fc2 = nn.Linear(50, 50)
		# self.fc3 = nn.Linear(50, 10)

	def forward(self, x):
		x = x.view(-1, 3*32*32)
		
		# F.sigmoid can be replaced with F.relu for Part Two
		x = F.sigmoid(self.fc1(x))
		
		# One hidden layer
		x = self.fc2(x)
		
		# Two hidden layers
		# x = F.sigmoid(self.fc2(x))
		# x = self.fc3(x)
		
		return x

def main():
	print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)	
	
	train_loader, validation_loader = get_data()
	for X_train, y_train in train_loader:
		print('X_train:', X_train.size(), 'type:', X_train.type())
		print('y_train:', y_train.size(), 'type:', y_train.type())
		break
			
	print(train_loader.dataset)
	print(validation_loader.dataset)
	
	# Part One & Two - Change Activation Function in Net Object
	for i in xrange(4):
		cte(train_loader, validation_loader, F.nll_loss, 0.01*pow(10, -i), 0.9, 5)
		
	# Part Three
	# Play with cte using different drop out, momentum, and weight decay.
	# Try to maximize accuracy / minimize loss
	
	# Part Four
	# Comment out the One hidden layer code, uncomment Two hidden layers
	# Test the same as in parts One & Two
	

def cte(train_loader, validation_loader, loss_function, learn_rate, momentum, epochs = 10):
	net = Net()
	if cuda:
		net.cuda()
	optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=momentum)
	for epoch in range(epochs):
		train(net, train_loader, optimizer, epoch, loss_function)
		validate(net, validation_loader, loss_function)	
	
	correct = 0
	total = 0
	with torch.no_grad():
		for data in validation_loader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))
	
def train(net, train_loader, optimizer, epoch, loss_function, log_interval=100):
	net.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = net(data)
		loss = F.cross_entropy(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
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

	accuracy = 100. * correct / len(validation_loader.dataset)
	#accuracy_vector.append(accuracy)
	
	print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			val_loss, correct, len(validation_loader.dataset), accuracy))

def get_data():
	transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	batch_size = 10
	kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

	train_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10('./data', train=True, download=True, transform=transform),
		batch_size=batch_size, shuffle=True, **kwargs)

	validation_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10('./data', train=False, transform=transform),
		batch_size=batch_size, shuffle=False, **kwargs)
	
	return train_loader, validation_loader

if __name__ == "__main__":
	main()