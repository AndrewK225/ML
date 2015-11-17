from sklearn import svm
import numpy as np
from numpy import genfromtxt,savetxt
from sklearn.ensemble import RandomForestClassifier

def main():
    feats = genfromtxt(open('train_no_nan.csv','r'),delimiter = ",",skip_header=1,usecols=(1,4,6))
    labels = genfromtxt(open('train_no_nan.csv','r'),delimiter = ",",skip_header=1,usecols=(0))
    test = genfromtxt(open('test.csv','r'),delimiter = ',',skip_header=1,usecols=(0,3,5))

    feats = np.nan_to_num(feats) 
    labels = np.nan_to_num(labels)
    test = np.nan_to_num(test)
    # Going to use random forests in order to get class probabilities
    rf = RandomForestClassifier(n_estimators=10,n_jobs=-1)
    
    rf.fit(feats,labels)
    savetxt('t1.csv',rf.predict_proba(test),header='"VisitNumber","TripType_3","TripType_4","TripType_5","TripType_6","TripType_7","TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18","TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24","TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30","TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36","TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42","TripType_43","TripType_44","TripType_999"'
            ,delimiter=',',fmt='%0.4f')
    
if __name__ == "__main__":
    main()
