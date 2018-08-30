import sys, os
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D, AveragePooling1D
from keras import backend as K
from sklearn.neighbors import NearestNeighbors

# During training used a batch size of 200
batch_size = 200
# Trained our networks for up to 10 iterations
#epochs = 10
epochs = 1

# We conducted a full grid search of the following attributes of both our CNN architecture and input/output format:
# the dimensionality of our convolutions (1D or 2D),
# the kernel size (i.e. the width of our 1D convolutional filters and both
# the height and width of our square 2D filters;
# we tried each multiple of 2 ranging from 2 to 10),
# whether to include dropout (yes or no) following max pooling steps or dense layers,
# whether to sort our rows based on similarity (yes or no),
# whether to log-scale our response variables (yes or no),
# and whether to represent ancestral and derived alleles as -1/1 or as 0/255.
convDim, convSize, poolSize, useLog, useInt, sortRows, useDropout, lossThreshold, inDir, weightFileName, modFileName, testPredFileName = sys.argv[1:]
convDim = convDim.lower()
convSize, poolSize = int(convSize), int(poolSize)
useLog = True if useLog.lower() in ["true","logy"] else False
useInt = True if useInt.lower() in ["true","intallele"] else False
sortRows = True if sortRows.lower() in ["true","sortrows"] else False
useDropout = True if useDropout.lower() in ["true","dropout"] else False
lossThreshold = float(lossThreshold)

def resort_min_diff(amat):
    ###assumes your snp matrix is indv. on rows, snps on cols
    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

X = []
y = []
print("reading data")
# The input is an alignment represented as an image.
# The input data is an alignment of linked segregating sites with partially shared evolutionary histories.
for npzFileName in os.listdir(inDir):
    if npzFileName.endswith(".npz"):
        u = np.load(inDir + npzFileName)
        currX, curry = [u[i] for i in  ('X', 'y')]
        ni,nr,nc = currX.shape
        newCurrX = []
        for i in range(ni):
            currCurrX = [currX[i,0]]
            if sortRows:
                currCurrX.extend(resort_min_diff(currX[i,1:]))
            else:
                currCurrX.extend(currX[i,1:])
            currCurrX = np.array(currCurrX)
            newCurrX.append(currCurrX.T)
        currX = np.array(newCurrX)
        assert currX.shape == (ni,nc,nr)
        #indices = [i for i in range(nc) if i % 10 == 0]
        #X.extend(np.take(currX,indices,axis=1))
        X.extend(currX)
        y.extend(curry)
        #if len(y) == 10000:
        #    break

y = np.array(y)
numParams=y.shape[1]
if useLog:
    y[y == 0] = 1e-6#give zeros a very tiny value so they don't break our log scaling
    y = np.log(y)
totInstances = len(X)
#testSize=10000
#valSize=10000
testSize=100
valSize=100
print("formatting data arrays")
X = np.array(X)
posX=X[:,:,0]
X=X[:,:,1:]
# positional information is a vector whose length is the maximum of the number of segregating sites
# observed across all simulated examples minus one.
imgRows, imgCols = X.shape[1:]

if useInt:
    X = X.astype('int8')
else:
    X = X.astype('float32')/127.5-1 # what is this?
if convDim == "2d":
    X = X.reshape(X.shape[0], imgRows, imgCols, 1).astype('float32')
posX = posX.astype('float32')/127.5-1

assert totInstances > testSize+valSize

testy=y[:testSize]
valy=y[testSize:testSize+valSize]
y=y[testSize+valSize:]
testX=X[:testSize]
testPosX=posX[:testSize]
valX=X[testSize:testSize+valSize]
valPosX=posX[testSize:testSize+valSize]
X=X[testSize+valSize:]
posX=posX[testSize+valSize:]

yMeans=np.mean(y, axis=0)
yStds=np.std(y, axis=0)
y = (y-yMeans)/yStds
testy = (testy-yMeans)/yStds
valy = (valy-yMeans)/yStds

print(len(X), len(y), len(yMeans))
print(yMeans, yStds)
print(X.shape, testX.shape, valX.shape)
print(posX.shape, testPosX.shape, valPosX.shape)
print(y.shape, valy.shape)
print("ready to learn (%d params, %d training examples, %d rows, %d cols)" %(numParams, len(X), imgRows, imgCols))

if convDim == "2d":
    # A 2-dimensional (2D) convolutional filter, which is more often used with image data,
    # allows the user to specify both dimensions of the filter matrix (often using a square matrix).
    inputShape = (imgRows, imgCols, 1)
    convFunc = Conv2D
    poolFunc = MaxPooling2D
else:
    # 1-dimensional convolutions are often used in the application to time-series data,
    # but are also applicable to sequence alignment matrices.
    # 1D filter is a rectangular matrix that spans a user-defined number of entries (called the "kernel size")
    # in one dimension in the input data (in our case this dimension is that of the polymorphic sites in the alignment),
    # and stretches entirely across the other dimension (in our case across all chromosomes in the sample).
    inputShape = (imgRows, imgCols)
    convFunc = Conv1D
    poolFunc = AveragePooling1D

# The image is passed through a first convolutional layer in order to create a set of feature maps.
# First convolutional layer, producing 128 filters
b1 = Input(shape=inputShape)
conv11 = convFunc(128, kernel_size=convSize, activation='relu')(b1)
# The set of feature maps are downsized via a pooling step.
# Max pooling layer with a kernel size given as arg - in paper poolSize=2.
pool11 = poolFunc(pool_size=poolSize)(conv11)
if useDropout:
    # dropout layers immediately follow max pooling steps.
    pool11 = Dropout(0.25)(pool11)

# Second convolutional layer, producing 128 filters
# These feature maps are then passed through a second convolutional filter and pooling step.
conv12 = convFunc(128, kernel_size=2, activation='relu')(pool11)
# The set of feature maps are downsized via a pooling step.
# Max pooling layer with a kernel size given as arg - in paper poolSize=2.
pool12 = poolFunc(pool_size=poolSize)(conv12)
if useDropout:
    # dropout layers immediately follow max pooling steps.
    # dropout step randomly removes 25% of neurons.
    pool12 = Dropout(0.25)(pool12)

# Third convolutional layer, producing 128 filters
# These feature maps are then passed through a third convolutional filter and pooling step.
conv13 = convFunc(128, kernel_size=2, activation='relu')(pool12)
# The set of feature maps are downsized via a pooling step.
# Max pooling layer with a kernel size given as arg - in paper poolSize=2.
pool13 = poolFunc(pool_size=poolSize)(conv13)
if useDropout:
    # dropout layers immediately follow max pooling steps.
    # dropout step randomly removes 25% of neurons.
    pool13 = Dropout(0.25)(pool13)

# Fourth convolutional layer, producing 128 filters
conv14 = convFunc(128, kernel_size=2, activation='relu')(pool13)
# The set of feature maps are downsized via a pooling step.
# Max pooling layer with a kernel size given as arg - in paper poolSize=2.
pool14 = poolFunc(pool_size=poolSize)(conv14)
if useDropout:
    # dropout layers immediately follow max pooling steps.
    # dropout step randomly removes 25% of neurons.
    pool14 = Dropout(0.25)(pool14)
# The resulting output is flattened in order to be passed as input into a fully connected feedforward layer.
flat11 = Flatten()(pool14)

# dense neural network layer (consisting of 32 nodes) taking positional information as its input
b2 = Input(shape=(imgRows,))
dense21 = Dense(32, activation='relu')(b2)
if useDropout:
    # dropout layers immediately follow the dense layer.
    # dropout step randomly removes 25% of neurons.
    dense21 = Dropout(0.25)(dense21)

# concatenate dense neural network layer output with that of the final max pooling layer of the CNN
merged = concatenate([flat11, dense21])
# feed concatenated dense neural net output and max pooling layer into final dense layer (256 nodes).
denseMerged = Dense(256, activation='relu', kernel_initializer='normal')(merged)
if useDropout:
    # dropout layers immediately follows the final dense layer.
    # dropout step randomly removes 25% of neurons.
    denseMerged = Dropout(0.25)(denseMerged)
denseOutput = Dense(numParams)(denseMerged)
model = Model(inputs=[b1, b2], outputs=denseOutput)
print(model.summary())

model.compile(loss='mean_squared_error', optimizer='adam')
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
checkpoint = keras.callbacks.ModelCheckpoint(weightFileName, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [earlystop, checkpoint]

# During training we used a batch size of 200.
# Training continues for a number of iterations (called epochs) until a specified stopping criterion is reached,
# up to 10 iterations, and
# retained the best performing CNN as assessed on the validation set.
model.fit([X, posX], y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([valX, valPosX], valy), callbacks=callbacks)


# Evaluted the performance of the best CNN on the test set by calculating total RMSE
#now we load the weights for the best-performing model
model.load_weights(weightFileName)
model.compile(loss='mean_squared_error', optimizer='adam')

#now we get the loss for our best model on the test set and emit predictions
testLoss = model.evaluate([testX, testPosX], testy)
print(testLoss)
preds = model.predict([testX, testPosX])
with open(testPredFileName, "w") as outFile:
    for i in range(len(preds)):
        outStr = []
        for j in range(len(preds[i])):
            outStr.append("%f vs %f" %(testy[i][j], preds[i][j]))
        outFile.write("\t".join(outStr) + "\n")

#if the loss is lower than our threshold we save the model file if desired
if modFileName.lower() != "nomod" and testLoss <= lossThreshold:
    with open(modFileName, "w") as modFile:
        modFile.write(model.to_json())
else:
    os.system("rm %s" %(weightFileName))
