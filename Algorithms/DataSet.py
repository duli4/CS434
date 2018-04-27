import numpy as np

class DataSet:
	def __init__(self, input_file_path, set_type, delim = None, y_col = 0):
		input_file = open(input_file_path, 'r')
		raw_data = [[float(val) for val in row.split(delim)] for row in input_file.read().splitlines()]
		
		if set_type == 'lin_reg':
			self.data_lx = [[1] + row[:y_col] + row[y_col+1:] for row in raw_data]
			self.data_ly = [row[y_col] for row in raw_data]
		elif set_type == 'per':
			self.data_lx = [row[:y_col] + row[y_col+1:] for row in raw_data]
			self.data_ly = [1.0 if row[y_col] > 0 else -1.0 for row in raw_data]
		elif set_type == 'log_reg':
			self.data_lx = [row[:y_col] + row[y_col+1:] for row in raw_data]
			self.data_ly = [float(row[y_col]) for row in raw_data]
		elif set_type == 'knn':
			self.data_lx = [row[:y_col] + row[y_col+1:] for row in raw_data]
			self.data_ly = [float(row[y_col]) for row in raw_data]
		elif set_type == 'dt':
			self.data_lx = [row[:y_col] + row[y_col+1:] for row in raw_data]
			self.data_ly = [float(row[y_col]) for row in raw_data]
			self.data_lm = raw_data
		
		self.data_mx = normalize_matrix(np.matrix(self.data_lx))
		self.data_my = np.matrix(self.data_ly).transpose()
		
		input_file.close()		

def get_sse(mg, my):
	return sum(map(lambda v: v**2, ([my[i] - mg[i] for i in xrange(np.shape(my)[0])])))
		
def get_ase(mg, my):
	return (get_sse(mg, my) / np.shape(my)[0])

def get_highest_factor(min, max, num):
	for i in range(min, max+1, 1):
		if (num % i == 0):
			hf = i
			
	return hf
	
def normalize_matrix(m):
	ranges = [(np.min(c), np.max(c)) for c in m.transpose()]
	lm = m.transpose().tolist()
	for i in xrange(len(lm)):
		for j in xrange(len(lm[i])):
			lm[i][j] = normalize_point(ranges[i][0], ranges[i][1], lm[i][j])
	
	return np.matrix(lm).transpose()
	
def normalize_point(min, max, val):
	return (val-min)/(max-min)