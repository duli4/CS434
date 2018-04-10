import numpy as np
import scipy as sp

def load_X_and_Y(filename,features,rows):
   f = open(filename,"r")
   all_data = f.read().splitlines()
   f.close()
   X = np.zeros( (rows,features + 1) )
   Y = np.zeros( (rows,1) )
   for i in xrange(0,rows):
      data = all_data[i].split()
      for x in xrange(0,features):
	 X[i][x] = float(data[x])
      Y[i][0] = float(data[features])
      X[i][features] = 1 #dummy variable
   return (X,Y)

def calculate_w(X,Y):
   X_T = np.transpose(X)
   X_T_X_inv = np.linalg.inv(np.matmul(X_T,X))
   w = np.matmul(np.matmul(X_T_X_inv,X_T),Y)
   return w

def main():
   #1.1: Load X and Y, calculate w and report it
   (X,Y) = load_X_and_Y("../housing_train.txt",13,433)
   w = calculate_w(X,Y)
   print "w:"
   print w

   #1.2: Apply learned weight vector to training data and testing data, respectively, and compute and report their ASEs

   #1.3: Remove the dummy variable from X, repeat 2 and 3

   #1.4: Modify the data by adding random features
if __name__ == "__main__": main()
