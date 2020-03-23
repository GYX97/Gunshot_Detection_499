import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer
import pickle
import get_data as gd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from adacost import AdaCostClassifier 
from LSTM import acu_curve


def cross_validation(scaled_X,y,classifier,K_num=2):
    '''
    Doing cross-validation to get the split of training and testing data with the best performance.
    The spliting range from 2 to 9
    '''
    cross_scores = []
    if classifier =='knn':
        knn = KNeighborsClassifier(n_neighbors=K_num)
        for i in range(2,10):
            scores = cross_val_score(knn,scaled_X,y,cv =i )
            cross_scores.append(scores.mean())      #Record the scores of cross validation
        plt.plot(range(2,10),cross_scores)
        plt.show()
    elif classifier =='adacost':
        Adacost=AdaCostClassifier(n_estimators=100)
        for i in range(2,10):
            scores = cross_val_score(Adacost,scaled_X,y,cv =i )
            cross_scores.append(scores.mean())
        plt.plot(range(2,10),cross_scores)
        plt.show()
    elif classifier =='svm':
        svm=LinearSVC()
        for i in range(2,10):
            scores = cross_val_score(svm,scaled_X,y,cv =i )
            cross_scores.append(scores.mean())
        # plt.plot(range(2,10),cross_scores)
        # plt.show()
    return cross_scores.index(max(cross_scores))

def data_pre():
    '''
    Get the data and stack all the data together with input data and labels
    '''
    print('extracting gunshot&ngunshot features...')
    x_Gunshot,s_Gunshot,x_nGunshot,s_nGunshot,dataset,As_Gunshot= gd.extract_features()
    print('gunshot&ngunshot features extracted!')
    X = np.vstack((s_Gunshot, s_nGunshot)).astype(np.float64)
    X = X.reshape(-1,22*394)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    As_Gunshot = np.array(As_Gunshot)
    As_Gunshot = As_Gunshot.reshape(-1,22*394)
    # Fit a per-column scaler
    As_Gunshot1= StandardScaler().fit(As_Gunshot)
    # Apply the scaler to X
    As_Gunshot = As_Gunshot1.transform(As_Gunshot)
    # # Define the labels vector
    y = np.hstack((np.ones(len(s_Gunshot)), -np.ones(len(s_nGunshot))))
    y_A = np.ones(len(As_Gunshot))
    return scaled_X, y, As_Gunshot,y_A

def data_split(scaled_X,y,split=3):
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=1/(split+2), random_state=rand_state)
    
    return X_train, X_test, y_train, y_test

    print('Feature vector length:', len(X_train[0]))

def knn_test(X_train, X_test, y_train, y_test):
    '''
    Test the KNN model to decide the best k which represents the number of the nearest neighbor
    '''
    error_rate = []
    for i in range(1,10):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    plt.figure(figsize=(10,6))
    plt.plot(range(1,10),error_rate,color = 'blue',linestyle = 'dashed',marker = 'o',
            markerfacecolor = 'red' , markersize = 10)
    plt.show()
    return error_rate.index(min(error_rate))+1


def knn_model(X_train, X_test, y_train, y_test, K_num):
    # set the sample size

    knn=KNeighborsClassifier(n_neighbors = K_num)
    # Check the training time for the KNN
    t=time.time()
    knn.fit(X_train,y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train KNN...')
    # Check the score of the KNN
    print('Test Accuracy of KNN = ', round(knn.score(X_test, y_test), 4))
    t0 = timer()
    y_pred = knn.predict(X_test)
    print("done in %0.3fs" % (timer() - t0))
    target_names = ['not_gunshot','gunshot']
    print(classification_report(y_test, y_pred,target_names=target_names))
    print(confusion_matrix(y_test, y_pred))
    acu_curve(y_pred,y_test)
    # Check the prediction time for a single sample
    # save classifier
    clf_pickle = {}
    clf_pickle["knn"] = knn

    destnation = 'gun_knn.p'
    pickle.dump( clf_pickle, open( destnation, "wb" ) )
    print("Classifier is written into: {}".format(destnation))

def svm_model(X_train, X_test, y_train, y_test):

    svc=LinearSVC()
    # Check the training time for the SVM
    t=time.time()
    svc.fit(X_train,y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVM...')
    # Check the score of the SVM
    print('Test Accuracy of SVM = ', round(svc.score(X_test, y_test), 4))
    t0 = timer()
    y_pred = svc.predict(X_test)
    print("done in %0.3fs" % (timer() - t0))
    target_names = ['not_gunshot','gunshot']
    print(classification_report(y_test, y_pred,target_names=target_names))
    print(confusion_matrix(y_test, y_pred))
    acu_curve(y_pred,y_test)
    # Check the prediction time for a single sample
    # save classifier
    clf_pickle = {}
    clf_pickle["svm"] = svc
    #clf_pickle["scaler"] = scaled_X

    destnation = 'gun_SVM.p'
    pickle.dump( clf_pickle, open( destnation, "wb" ) )
    print("Classifier is written into: {}".format(destnation))

def Adacost(X_train,X_test,y_train,y_test):
    Adacost = AdaCostClassifier(n_estimators=100)
    t=time.time()
    Adacost.fit(X_train,y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train Adacost...')
    # Check the score of the Adacost
    print('Test Accuracy of Adacost = ', round(Adacost.score(X_test, y_test), 4))
    t0 = timer()
    y_pred = Adacost.predict(X_test)
    print("done in %0.3fs" % (timer() - t0))
    target_names = ['not_gunshot','gunshot']
    print(classification_report(y_test, y_pred, labels = [-1,1],target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=[-1,1]))
    acu_curve(y_pred,y_test)
if __name__ == "__main__":
    scaled_X,y, As_Gunshot, y_A = data_pre()
    
    
    #split = cross_validation(scaled_X,y,classifier = 'adacost')
    X_train, X_test, y_train, y_test = data_split(scaled_X,y)
    #K_num = knn_test(X_train, X_test, y_train, y_test)
    # X_test = As_Gunshot
    # y_test = np.ones(len(X_test))
    svm_model(X_train, X_test, y_train, y_test)
    #Adacost(X_train, X_test, y_train, y_test)