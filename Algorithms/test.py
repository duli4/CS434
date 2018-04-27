import DataSet as ds
import ml_algs as mla
import numpy as np

def main():
	# test_lin_reg()
	# test_per('batch')
	# test_per('online')
	# test_per('voted')
	#test_log_reg('batch')
	#test_log_reg('online')
	test_knn()

def test_lin_reg():
	trn = ds.DataSet("./Data/housing_train.txt", set_type='lin_reg', y_col=13)
	tst = ds.DataSet("./Data/housing_test.txt", set_type='lin_reg', y_col=13)
	
	w = mla.get_data_mwlin(trn.data_mx, trn.data_my)
	
	trn_g = mla.get_data_mglin(w, trn.data_mx)
	trn_sse = ds.get_sse(trn_g, trn.data_my)
	trn_ase = ds.get_ase(trn_g, trn.data_my)
	
	tst_g = mla.get_data_mglin(w, tst.data_mx)
	tst_sse = ds.get_sse(tst_g, tst.data_my)
	tst_ase = ds.get_ase(tst_g, tst.data_my)
	
	#print "W: ", w
	#print "Training G: ", trn_g
	#print "Testing G: ", tst_g
	
	print "Linear Regression: "
	print "Training SSE: ", trn_sse
	print "Testing SSE: ", tst_sse
	print "Training ASE: ", trn_ase
	print "Testing ASE: ", tst_ase, "\n"
	
def test_per(type = 'batch'):
	trn = ds.DataSet("./Data/usps_train.csv", set_type='per', delim = ',', y_col=256)
	tst = ds.DataSet("./Data/usps_test.csv", set_type='per', delim = ',', y_col=256)
	
	if type == 'batch':
		w = mla.get_data_mwpb(trn.data_mx, trn.data_my)
	elif type == 'online':
		w = mla.get_data_mwpo(trn.data_mx, trn.data_my)
	elif type == 'voted':
		w = mla.get_data_mwpv(trn.data_mx, trn.data_my)
	
	trn_g = mla.get_data_mgp(w, trn.data_mx)
	trn_sse = ds.get_sse(trn_g, trn.data_my)
	trn_ase = ds.get_ase(trn_g, trn.data_my)
	
	tst_g = mla.get_data_mgp(w, tst.data_mx)
	tst_sse = ds.get_sse(tst_g, tst.data_my)
	tst_ase = ds.get_ase(tst_g, tst.data_my)
	
	# print "WS: ", np.shape(w)
	# print "XS: ", np.shape(trn.data_mx)
	# print "YS: ", np.shape(trn.data_my)
	
	
	print type.upper(), " Perceptron: "
	print "Training Mistakes: ", trn_sse
	print "Testing Mistakes:: ", tst_sse
	print "Training Mistake Percent: ", trn_ase
	print "Testing Mistake Percent: ", tst_ase, "\n"

def test_log_reg(type = 'batch'):
	trn = ds.DataSet("./Data/usps_train.csv", set_type='log_reg', delim = ',', y_col=256)
	tst = ds.DataSet("./Data/usps_test.csv", set_type='log_reg', delim = ',', y_col=256)
	
	if type == 'batch':
		w = mla.get_data_mwlogb(trn.data_mx, trn.data_my)
	elif type == 'online':
		w = mla.get_data_mwlogo(trn.data_mx, trn.data_my)
	
	trn_g = mla.get_data_mglog(w, trn.data_mx)
	trn_sse = ds.get_sse(trn_g, trn.data_my)
	trn_ase = ds.get_ase(trn_g, trn.data_my)
	
	tst_g = mla.get_data_mglog(w, tst.data_mx)
	tst_sse = ds.get_sse(tst_g, tst.data_my)
	tst_ase = ds.get_ase(tst_g, tst.data_my)
	
	#print "W: ", w
	#print "Training G: ", trn_g
	#print "Testing G: ", tst_g
	
	print "Logistic Regression: "
	print "Training SSE: ", trn_sse
	print "Testing SSE: ", tst_sse
	print "Training ASE: ", trn_ase
	print "Testing ASE: ", tst_ase, "\n"
	
def test_knn():
	trn = ds.DataSet("./Data/knn_train.csv", set_type='knn', delim = ',', y_col=0)
	tst = ds.DataSet("./Data/knn_test.csv", set_type='knn', delim = ',', y_col=0)

def test_dt():
	trn = ds.DataSet("./Data/knn_train.csv", set_type='dt', delim = ',', y_col=0)
	tst = ds.DataSet("./Data/knn_test.csv", set_type='dt', delim = ',', y_col=0)
	
if __name__ == "__main__":
	main()