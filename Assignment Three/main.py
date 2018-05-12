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
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def main():
	print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)	
	
	train_loader, validation_loader = get_data('CIFAR10')
	for (X_train, y_train) in train_loader:
			print('X_train:', X_train.size(), 'type:', X_train.type())
			print('y_train:', y_train.size(), 'type:', y_train.type())
			break
			
	model = Net()
	if cuda:
		model.cuda()
	
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
	print(model)
	
	correct = 0
	total = 0
	with torch.no_grad():
		for data in validation_loader:
			images, labels = data
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))
	
	
	# epochs = 10

	# lossv, accv = [], []
	# for epoch in range(1, epochs + 1):
			# train(model, train_loader, optimizer, epoch)
			# validate(model, validation_loader)
	
def train(model, train_loader, optimizer, epoch, log_interval=100):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))
				
def validate(model, validation_loader):
	model.eval()
	val_loss, correct = 0, 0
	for data, target in validation_loader:
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		val_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	#loss_vector.append(val_loss)

	accuracy = 100. * correct / len(validation_loader.dataset)
	#accuracy_vector.append(accuracy)
	
	print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			val_loss, correct, len(validation_loader.dataset), accuracy))

def get_data(set_name = 'MNIST'):
	batch_size = 32
	kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

	if set_name == 'CIFAR10':
		train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10('./data', train=True, download=True,
										 transform=transforms.Compose([
												 transforms.ToTensor(),
												 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
										 ])),
			batch_size=batch_size, shuffle=True, **kwargs)

		validation_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
												 transforms.ToTensor(),
												 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
										 ])),
			batch_size=batch_size, shuffle=False, **kwargs)
	elif set_name == 'MNIST':
		train_loader = torch.utils.data.DataLoader(
			datasets.MNIST('./data', train=True, download=True,
									 transform=transforms.Compose([
											 transforms.ToTensor(),
											 transforms.Normalize((0.1307,), (0.3081,))
									 ])),
		batch_size=batch_size, shuffle=True, **kwargs)

	validation_loader = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train=False, transform=transforms.Compose([
											 transforms.ToTensor(),
											 transforms.Normalize((0.1307,), (0.3081,))
									 ])),
		batch_size=batch_size, shuffle=False, **kwargs)
	return (train_loader, validation_loader)

if __name__ == "__main__":
	main()