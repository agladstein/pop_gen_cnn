import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Merge
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras import backend as K
from random import shuffle, choice
from keras.preprocessing import sequence
from matplotlib import pyplot as plt


def rsquare(x,y):
    return np.corrcoef(x,y)[0][1]**2  #r-squared
def rmse(x,y):
    return np.sqrt(np.mean((x-y)**2))


a = np.load('./dataset/autotet.ld.data.npz' , encoding='latin1') #made npz in py2, had to do this to read in py3 training env.

postrain, ytest, ytrain, xtest, xtrain, postest = [a[i] for i in ['postrain', 'ytest', 'ytrain', 'xtest', 'xtrain', 'postest']]

#find max matrix
m = 0 
for i in xtrain:
    if i.shape[0] > m: m = i.shape[0]
for i in xtest:
    if i.shape[0] > m: m = i.shape[0]

#pad
xtrain = sequence.pad_sequences(xtrain, maxlen=m, padding='post', dtype='float32')
xtest = sequence.pad_sequences(xtest, maxlen=m, padding='post', dtype='float32')

postrain = sequence.pad_sequences(postrain, maxlen=m, padding='post', value=-1., dtype='float32')
postest = sequence.pad_sequences(postest, maxlen=m, padding='post', value=-1., dtype='float32')


#clean y data
ytrain_rho, ytest_rho = np.array([i[1] for i in ytrain]), np.array([i[1] for i in ytest])
#ytrain_rho_over_theta, ytest_rho_over_theta = np.array([i[1]/i[0] for i in ytrain]), np.array([i[1]/i[0] for i in ytest])

mean_test = np.mean(np.log(ytest_rho))
mean_train = np.mean(np.log(ytrain_rho))

ytrain_rho_log_centered = np.log(ytrain_rho) - mean_train
plt.hist(ytrain_rho_log_centered)
plt.show()

ytest_rho_log_centered = np.log(ytest_rho) - mean_test
plt.hist(ytest_rho_log_centered)
plt.show()
print ("mean of training data, save:", mean_train) #the mean train is important, it's 4.78838995038


#data preped, now build model
#b1 is conv branch, b2 is position data branch
b1 = Sequential()
b1.add(Conv1D(512, kernel_size=2,
                 activation='relu',
                 input_shape=(xtest.shape[1], xtest.shape[2])))
b1.add(Conv1D(256, kernel_size=2, activation='relu'))
b1.add(AveragePooling1D(pool_size=2))
b1.add(Dropout(0.25))
b1.add(Conv1D(256, kernel_size=2, activation='relu'))
b1.add(AveragePooling1D(pool_size=2))
b1.add(Dropout(0.25))
b1.add(Flatten())

b2 = Sequential()
b2.add(Dense(64, input_shape = (460,), activation='relu'))
b2.add(Dropout(0.25))

model = Sequential()
#model.add([b1, b2])
#model = keras.layers.concatenate([b1,b2])
model.add(Merge([b1, b2], mode = 'concat'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

#train!  this one trains fast
batch_size = 64
for epoch in range(3):
    if epoch < 1: model.summary()

    model.fit([xtrain, postrain], ytrain_rho_log_centered, batch_size=batch_size,
              epochs=1,
              verbose=1,
              validation_data=([xtest, postest], ytest_rho_log_centered))
    pred = [i[0] for i in model.predict([xtest, postest])]
    plt.scatter(np.exp(ytest_rho_log_centered+mean_train), np.exp(pred+mean_train), alpha=.4)
    plt.semilogy()
    plt.semilogx()
    plt.show()
    print (epoch, rsquare(np.exp(ytest_rho_log_centered+mean_train), np.exp(pred)+mean_train), 
           rmse(np.exp(ytest_rho_log_centered+mean_train), np.exp(pred+mean_train)))

#done, save that json string and the weights file.  need those for validation
j = model.to_json()
print(j)
model.save_weights('3rd.autotet.mergnet.weights')

#from matplotlib import pyplot as plt
#pred = np.array([np.exp(ii[0]+mean_test) for ii in model.predict([xtest,postest])])
#print(rmse(pred, ytest_rho))
#thetas=[i[0] for i in ytest]

#print(map(len, (thetas, pred, ytest_rho)))

#q = {}
#for i,j,k in zip(thetas, pred, ytest_rho):
#    if i not in q: q[i] = []
#    q[i].append((j,k))
#for i in q: print( i, len(q[i]) )

#idx=1
#k1, k2 = [],[]
#for i in sorted(q):
#    plt.subplot(2,3,idx)
#    pred, real = [n[0] for n in q[i]], [n[1] for n in q[i]]
#    plt.scatter(real, pred, alpha=.3)
#    r = rsquare(np.array(real), np.array(pred))
#    plt.title('theta = '+str(i)+' r2:'+str(round(r,1)))
#    idx+=1
#    k1.extend(real)
#    k2.extend(pred)
#    print (i, r)
#plt.show()


#d = []
#for i in sorted(q):
#    resid = [p-r for p,r in q[i]]
#    d.append(resid)
#plt.boxplot(d)
#plt.xticks(range(1,6), map(str, sorted(q)))
#plt.show()
#print(rsquare(np.array(k1), np.array(k2)))
