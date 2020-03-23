from Classifier import *
import numpy as np
from adacost import AdaCostClassifier 


def one_out(scaled_X,y,As_Gunshot,y_A):
    '''
    Leave one Africa data for testing and rest of data for training
    '''
    score = []
    for i in range(len(As_Gunshot)):
        test_x = As_Gunshot[i][np.newaxis,:]
        test_y = y_A[i]                         #The candidates to leave out
        X_train = np.delete(As_Gunshot,i,0)
        Y_train = np.delete(y_A,i)
        X = np.vstack((scaled_X, X_train)).astype(np.float64)   #Combine the rest Africa data with other data
        Y = np.hstack((y, Y_train)).astype(np.float64)
        #split = cross_validation(X,Y,classifier = 'svm')
        X_train, X_test, y_train, y_test = data_split(X,Y)
        Adacost = AdaCostClassifier(n_estimators=100)
        Adacost.fit(X_train[:-10],y_train[:-10])
        test_x = np.vstack((test_x,X_train[-10:]))
        test_y = np.hstack((test_y,y_train[-10:]))      #Test on the chosen Africa data and 10 other data without trained
        print('Test Prediction of Adacost = ', Adacost.predict(test_x))
        for i in range(len(test_x)):
            if Adacost.predict(test_x[i][np.newaxis,:])==test_y[i]:
                score.append(1)
            else:
                score.append(0)
    print('The out put of one_out is', round(sum(score)/len(As_Gunshot)/11,4))      #Get the overall score

if __name__ == "__main__":
    scaled_X,y, As_Gunshot, y_A = data_pre()
    one_out(scaled_X,y,As_Gunshot,y_A)
