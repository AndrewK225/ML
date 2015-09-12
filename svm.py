#Support Vector Machines
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from numpy import genfromtxt,savetxt

def main():
    feats_all = genfromtxt(open('X_train.txt','r'))
    labels_all = genfromtxt(open('Y_train.txt','r'),dtype='int')
    test = genfromtxt(open('X_test.txt','r'))

    #train_feats = feats_all[:7000]
    #train_labels = labels_all[:7000]
    #test = feats_all[7000:]

    
    savetxt('svm_submission1.csv',OneVsRestClassifier(LinearSVC(random_state=18)).fit(feats_all,labels_all).predict(test),delimiter='\n')
    
    


if __name__=="__main__":
    main()
