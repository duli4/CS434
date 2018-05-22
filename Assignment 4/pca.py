import numpy as np
from PIL import Image

def main():
	data = load_data("unsupervised.txt")
	mm = data.mean(0)
	Image.fromarray(np.reshape(mm,(28,28)).astype('uint8'),mode='L').save('mean.png')
	cm = np.cov(data, rowvar=False)
	vals, vecs = np.linalg.eigh(cm)
	et = sorted([(float(vals[i]), vecs[i]) for i in xrange(len(vals))],key = lambda item:item[0],reverse=True)
	
	#PCA - Part One
	i = 0
	for (val,vec) in et[:10]:
	   max_val = max(vec)
	   modified_vec = list(map(lambda item:item*255.0/max_val,vec))
	   Image.fromarray(np.reshape(modified_vec,(28,28)).astype('uint8'),mode='L').save('eigen_'+str(i)+'.png')
	   i = i + 1
	print [et[i][0] for i in xrange(10)]
	
	
def load_data(filepath):
	return np.array([[float(item) for item in line.split(',')] for line in open(filepath, 'r').read().splitlines()])
	
if __name__ == "__main__":
	main()
