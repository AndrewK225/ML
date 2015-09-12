# Andrew Koe
# Machine Learning Kaggle
# 9/9/15

import pyreport
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt,savetxt

def main():
    feats_all = genfromtxt(open('X_train.txt','r'))
    labels_all = genfromtxt(open('Y_train.txt','r'),dtype='int')
    test = genfromtxt(open('X_test.txt','r'))

    #train_feats = feats_all[:7000]
    #train_labels = labels_all[:7000]
    #test = feats_all[7000:]

    rf = RandomForestClassifier(n_estimators=100,n_jobs=2)
    rf.fit(feats_all,labels_all)
    savetxt('sub_koe2.csv',rf.predict(test),delimiter='\n')
    
    
    
    


if __name__=="__main__":
    main()
