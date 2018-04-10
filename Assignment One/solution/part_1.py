import numpy as np
import scipy as sp

def load_X_and_Y(filename,features,rows,dummy):
   f = open(filename,"r")
   all_data = f.read().splitlines()
   f.close()
   X = np.zeros( (rows,features + (1 if dummy else 0)) )
   Y = np.zeros( (rows,1) )
   for i in xrange(0,rows):
      data = all_data[i].split()
      for x in xrange(0,features):
	 X[i][x] = float(data[x])
      Y[i][0] = float(data[features])
      if dummy:
      	X[i][features] = 1 #dummy variable
   return (X,Y)

def calculate_w(X,Y):
   X_T = np.transpose(X)
   X_T_X_inv = np.linalg.inv(np.matmul(X_T,X))
   w = np.matmul(np.matmul(X_T_X_inv,X_T),Y)
   return w

def compute_ASE(w,X,Y):
   SSE = 0
   num_features = len(w)
   num_points = len(X[0])
   #assume the lenght of X and Y is the same
   for i in xrange(0,num_points):
      guess = 0
      for x in xrange(0,num_features):
	 guess = guess + w[x][0]*X[i][x]
      SE = (Y[i][0]-guess)**2
      SSE = SSE + SE
   ASE = SSE/num_points
   return ASE
   

def main():
   #1.1: Load X and Y, calculate w and report it
   (X,Y) = load_X_and_Y("../housing_train.txt",13,433,True)
   w = calculate_w(X,Y)
   print "w:"
   print w

   #1.2: Apply learned weight vector to training data and testing data, respectively, and compute and report their ASEs
   ASE = compute_ASE(w,X,Y)
   print "ASE for training data:", ASE
   (X_test,Y_test) = load_X_and_Y("../housing_test.txt",13,74,True)
   ASE_test = compute_ASE(w,X_test,Y_test)
   print "ASE for testing data:", ASE_test
   #1.3: Remove the dummy variable from X, repeat 1 and 2

   #1.4: Modify the data by adding random features
if __name__ == "__main__": main()
