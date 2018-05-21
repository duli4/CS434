import numpy as np

def main():
	data = load_data("unsupervised.txt")
	mm = data.mean(0)
	cm = np.cov(data, rowvar=False)
	
	vals, vecs = np.linalg.eig(cm)
	et = [(vals[i], vecs[i]) for i in xrange(len(vals))]
	
	#PCA - Part One
	print [et[i][0] for i in xrange(10)]
	
	
def load_data(filepath):
	raw_data = [line.split(',') for line in open(filepath, 'r').read().splitlines()]
	return np.matrix(raw_data, dtype='int')
	
if __name__ == "__main__":
	main()