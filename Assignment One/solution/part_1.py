import numpy as np
import scipy as sp
import math

def load_X_and_Y(filename,features,rows,dummy,randoms=0):
   f = open(filename,"r")
   all_data = f.read().splitlines()
   f.close()
   X = np.zeros( (rows,features + (1 if dummy else 0) + randoms) )
   Y = np.zeros( (rows,1) )
   for i in xrange(0,rows):
      data = all_data[i].split()
      for x in xrange(0,features):
	 X[i][x] = float(data[x])
      Y[i][0] = float(data[features])
      if dummy:
      	X[i][features] = 1 #dummy variable
      for x in xrange(0,randoms):
	X[i][features + (1 if dummy else 0) + x] = np.random.normal()
   return (X,Y)

def calculate_w(X,Y):
   X_T = np.transpose(X)
   X_T_X_inv = np.linalg.inv(np.matmul(X_T,X))
   w = np.matmul(np.matmul(X_T_X_inv,X_T),Y)
   return w

def compute_ASE(w,X,Y):
   SSE = 0
   num_features = len(w)
   num_points = len(X)
   #assume the length of X and Y is the same (it should be, or you could not have gotten w!)
   for i in xrange(0,num_points):
      guess = 0
      for x in xrange(0,num_features):
	 guess = guess + w[x][0]*X[i][x]
      SE = math.pow((Y[i][0]-guess),2)
      SSE = SSE + SE
   ASE = SSE/num_points
   return ASE
   
def run_with_random(d):
   (X,Y) = load_X_and_Y("../housing_train.txt",13,433,True,d)
   (X_test,Y_test) = load_X_and_Y("../housing_test.txt",13,74,True,d)
   w = calculate_w(X,Y)
   ASE = compute_ASE(w,X,Y)
   ASE_test = compute_ASE(w,X_test,Y_test)
   return (ASE,ASE_test)

def main():
   #1.1: Load X and Y, calculate w and report it
   (X,Y) = load_X_and_Y("../housing_train.txt",13,433,True)
   w = calculate_w(X,Y)
   print "w:"
   print np.transpose(w)

   #1.2: Apply learned weight vector to training data and testing data, respectively, and compute and report their ASEs
   ASE = compute_ASE(w,X,Y)
   print "ASE for training data:", ASE
   (X_test,Y_test) = load_X_and_Y("../housing_test.txt",13,74,True)
   ASE_test = compute_ASE(w,X_test,Y_test)
   print "ASE for testing data:", ASE_test

   #1.3: Remove the dummy variable from X, repeat 1 and 2
   (X_dumb,Y_dumb) = load_X_and_Y("../housing_train.txt",13,433,False)
   w_dumb = calculate_w(X_dumb,Y_dumb)
   print "w without dummy variable: "
   print np.transpose(w_dumb)
   ASE_ = compute_ASE(w_dumb,X_dumb,Y_dumb)
   print "ASE for training data without dummy variable:", ASE_
   (X_dumb_test,Y_dumb_test) = load_X_and_Y("../housing_test.txt",13,74,False)
   ASE_test_ = compute_ASE(w_dumb,X_dumb_test,Y_dumb_test)
   print "ASE for testing data without dummy variable:", ASE_test_



   #1.4: Modify the data by adding random features
   test_vals = [2,4,6,8,10,20,50,100]
   for d in test_vals:
      (ASE_d,ASE_test_d) = run_with_random(d)
      print "d =", d, ":"
      print "Training ASE =", ASE_d
      print "Testing ASE =", ASE_test_d
if __name__ == "__main__": main()
