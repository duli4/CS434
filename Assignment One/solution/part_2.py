import numpy as np
import scipy as sp
import math

MU = 0.001
MyLambda = 128.0

def load_X_and_Y(filename,features,rows):
   f = open(filename,"r")
   all_data = f.read().splitlines()
   f.close()
   X = np.zeros( (rows,features+1) )
   Y = np.zeros( (rows,1) )
   for i in xrange(0,rows):
      data = all_data[i].split(',')
      for x in xrange(0,features):
	 X[i][x] = float(data[x])
      Y[i][0] = float(data[features])
      X[i][features] = 1 #dummy variable
   return (X,Y)

def calculate_w_BGD(X,Y,w,mu,Lambda):
   num_features = len(X[0])
   num_points = len(X)
   nabla = np.zeros( (num_features, 1) )
   for i in xrange(0,num_points):
      X_i = np.transpose(np.array([X[i]]))
      w_T_X_i = np.matmul(np.transpose(w),X_i)[0][0]
      if w_T_X_i < -700:
	 y_i_hat = 0.0
      else:
	 y_i_hat = 1.0/(1.0+math.exp(-w_T_X_i))
      nabla = nabla + (y_i_hat-Y[i][0])*X_i
   w = w - mu*(nabla + Lambda*w)
   nabla_norm = np.linalg.norm(nabla)
   return (w,nabla_norm)

def main():
   (X,Y) = load_X_and_Y("../usps-4-9-train.csv",256,1400)
   (X_test,Y_test) = load_X_and_Y("../usps-4-9-test.csv",256,800)
   w = np.zeros( (257, 1) )
   count = 0
   nabla_norm = calculate_w_BGD(X,Y,w,MU,MyLambda)
   last_correct = -1
   while nabla_norm > 0.5 and count < 250:
     (w,nabla_norm) = calculate_w_BGD(X,Y,w,MU,MyLambda)
     count = count + 1
     correct = 0
     for i in xrange(0,1400):
        X_i = np.transpose(np.array([X[i]]))
        w_T_X_i = np.matmul(np.transpose(w),X_i)[0][0]
	if w_T_X_i < -700: #avoid those overflows (e^709 overflows)
	   P = 0
	else:
	   P = 1.0/(1.0+math.exp(-w_T_X_i))

        guess = 1 if P > 0.5 else 0
        if guess == Y[i][0]:
	   correct = correct + 1
     training_accuracy = correct*100.0/1400.0
     correct = 0
     for i in xrange(0,800):
        X_i = np.transpose(np.array([X_test[i]]))
        w_T_X_i = np.matmul(np.transpose(w),X_i)[0][0]
	if w_T_X_i < -700: #avoid those overflows (e^709 overflows)
	   P = 0
	else:
	   P = 1.0/(1.0+math.exp(-w_T_X_i))

        guess = 1 if P > 0.5 else 0
        if guess == Y_test[i][0]:
	   correct = correct + 1
     test_accuracy = correct*100.0/800.0
     print "Iteration",count,":",training_accuracy,"% Training,",test_accuracy,"% Testing"
     last_correct = correct
   print "|nabla| =",nabla_norm

if __name__ == "__main__": main()
