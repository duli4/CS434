import numpy as np
from PIL import Image

def main():
	data = load_data("unsupervised.txt")
	
	mean = data.mean(0)
	Image.fromarray(np.reshape(mean,(28,28)).astype('uint8'),mode='L').save('mean.png')
	
	cov_mat = np.cov(data, rowvar=False)
	vals, vecs = np.linalg.eigh(cov_mat)
	vecs = np.transpose(vecs)
	
	et = sorted([(float(vals[i]), vecs[i]) for i in xrange(len(vals))],key = lambda item:item[0],reverse=True)
	# Part One
	# for i in xrange(10):
		# print "EigenValue " + str(i) + ": " + str(et[i][0])
	
	# Part Two
	# for i, (val,vec) in enumerate(et[:10]):
	   # max_val = max(vec)
	   # modified_vec = list(map(lambda item:item*255.0/max_val,vec))
	   # Image.fromarray(np.reshape(modified_vec,(28,28)).astype('uint8'),mode='L').save('eigen_'+str(i)+'.png')
	   # i = i + 1
	
	# Part Three
	max_vals = [[-1,-100000] for i in xrange(10)]
	evs = [et[i][1] for i in xrange(10)]
	
	for i in xrange(len(data)):
		vals = np.dot(np.asmatrix(data[i]), np.asmatrix(evs).T)
		for j in xrange(10):
			if vals.item((0, j)) > max_vals[j][1]:
				max_vals[j][0] = i
				max_vals[j][1] = vals.item((0, j))
				
	for i, (idx, _) in enumerate(max_vals):
		Image.fromarray(np.reshape(data[idx],(28,28)).astype('uint8'),mode='L').save('image_'+str(i)+'.png')
			
			
	
def load_data(filepath):
	return np.array([[float(item) for item in line.split(',')] for line in open(filepath, 'r').read().splitlines()])
	
if __name__ == "__main__":
	main()
