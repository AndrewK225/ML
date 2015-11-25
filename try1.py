#!/home/cuee/anaconda/bin/python
import xgboost as xgb
import pandas as pd
import numpy as np
import math
from tempfile import TemporaryFile

def main():
	#38 trip types
	
	x_train,y_train,decoder = preprocess_training()
	x_test = preprocess_testing()

	y_test = [0]*x_test.size
	dtrain = xgb.DMatrix(x_train,label=y_train)
	dtest = xgb.DMatrix(x_test,label=np.array(y_test))
	
	param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'multi:softprob' ,'num_class':7}
	param['eval_metric'] = 'mlogloss'
	watchlist = [(dtest,'eval'),(dtrain,'train')]
	num_round = 2
	bst = xgb.train(param,dtrain,num_round,watchlist)
	bst.save_model('1.model')
	preds = bst.predict(dtest)
	
def preprocess_testing():
	test = pd.read_csv('test_debug.csv')
	uniq_fline = [] # No guarantee flines would be present in training
	features = {}
	end_features = []
	tmp ={}
	for num in test['FinelineNumber']:
		if not math.isnan(num) and num not in uniq_fline:
			uniq_fline.append(num)
	
	for visit in test['VisitNumber']:
		features[visit] = [0]*len(uniq_fline)
	
	sorted_fline = sorted(uniq_fline)

        #map each fline to a smaller number so it doesn't crash 
        for fline in sorted_fline:
                tmp[fline] = sorted_fline.index(fline)
		
	#binary vector for features
        for visit,fin in zip(test['VisitNumber'],test['FinelineNumber']):
                if not math.isnan(fin):
                        features[visit][tmp[fline]] += 1

	for key in sorted(features):
                end_features.append(features[key])

	feats = np.array(end_features,dtype=np.int)
	#test_feats_file = TemporaryFile()
	#np.save(test_feats_file,feats)
	return feats

def preprocess_training():
	train = pd.read_csv('train_debug.csv')
	uniq_fline = []
	features = {}
	labels = {}
	end_labels = []
	end_features = []
	tmp = {}
	uniq_types = []
	encode = {}
	decode = {}

	for num in train['FinelineNumber']:	
		if not math.isnan(num) and num not in uniq_fline:
			uniq_fline.append(num)
		
	for visit in train['VisitNumber']:
		features[visit] = [0]*len(uniq_fline)
	
	sorted_fline = sorted(uniq_fline)

	#map each fline to a smaller number so it doesn't crash	
	for fline in sorted_fline:
		tmp[fline] = sorted_fline.index(fline)
		
	#binary vector for features
	for visit,fin in zip(train['VisitNumber'],train['FinelineNumber']):
		if not math.isnan(fin):
			features[visit][tmp[fline]] += 1 	

	
	#labels
	#need to map labels to 0-38
	for tt in train['TripType']:
		if not math.isnan(tt) and tt not in uniq_types:
			uniq_types.append(tt)
	
	
	cnt = 0
	for tt in sorted(uniq_types):
		encode[tt] = cnt #tt to number
		decode[cnt] = tt #number to tt
		cnt += 1	
	  	
	for trip_type, visit in zip(train['TripType'],train['VisitNumber']):
		labels[visit] = encode[trip_type]
	
	for key in sorted(features):
		end_features.append(features[key])
		end_labels.append(labels[visit])

	#create the arrays of training data and save to disk
	feats = np.array(end_features,dtype=np.int)
	label = np.array(end_labels,dtype=np.int)
	#train_feats_file = TemporaryFile()
	#train_labels_file = TemporaryFile()
	#np.save(train_feats_file,feats)
	#np.save(train_labels_file,label)
	return feats,label,decode
if __name__ == "__main__":
	main()
