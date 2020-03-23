from keras.datasets import mnist
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout
from keras.utils import to_categorical
from keras.models import Sequential
import get_data as gd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def data_pre():
    '''
    Get the data and stack all the data together with input data and labels
    '''
    print('extracting gunshot&ngunshot features...')
    x_Gunshot,s_Gunshot,x_nGunshot,s_nGunshot,dataset,As_Gunshot= gd.extract_features()
    print('gunshot&ngunshot features extracted!')
    X = np.vstack((x_Gunshot, x_nGunshot)).astype(np.float64)
    length = len(X)
    X = np.reshape(X,(length,100,-1))
    # Fit a per-column scaler
    y = np.hstack((np.ones(len(x_Gunshot)), np.zeros(len(x_nGunshot))))
    return X, y

def build_model(lstm_layers,dense_layers):
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(nb_time_steps, nb_input_vector),return_sequences=True))
    for i in range(lstm_layers - 1):
            model.add(LSTM(output_dim=32 * (lstm_layers-i),
                        activation='relu',
                        return_sequences=True))
    model.add(LSTM(output_dim=32,
                        activation='relu',
                        return_sequences=False))
    for i in range(dense_layers - 1):
            model.add(Dense(output_dim=256,
                            activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def acu_curve(y,prob):
    fpr,tpr,threshold = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
 
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 40,
    }
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 50,
    }
    plt.xlabel('False Positive Rate',font1)
    plt.ylabel('True Positive Rate',font1)
    plt.title('Receiver operating characteristic curve',font2)
    plt.legend(loc="lower right")
 
    plt.show()


# parameters for LSTM
nb_lstm_outputs = 30  #Number of output
nb_time_steps = 100  #time steps
nb_input_vector = 1728 #input vector
lstm_layers = 3
dense_layers = 1

#data preprocessing: tofloat32, normalization, one_hot encoding
X, y = data_pre()
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

model = build_model(lstm_layers,dense_layers)
#compile:loss, optimizer, metrics


#train: epcoch, batch_size
model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=1)
model.summary()
score = model.evaluate(X_test, y_test,batch_size=64, verbose=1)

y_score = model.predict(X_test)
yt = []
yc = []
for i in range(len(y_test)):
    yt.append(np.where(y_test[i]==1)[0].tolist()[0])   
    yc.append(np.where(y_score[i]==1)[0].tolist()[0])
acu_curve(yt,yc)
target_names = ['not_gunshot','gunshot']
print(classification_report(yt, yc, target_names=target_names))
print(confusion_matrix(yt, yc))

print(score)



