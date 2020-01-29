import pandas as pd
#imports
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import math
import matplotlib.pyplot as plt
df = pd.read_excel(r"ML_TEST.xlsx")   
  #creating and plotting the data
#creating some data - we will create a sinus and a sinus * sinus function
shift = 7
leng = len(df)
Train_percent = .8
scaler = MinMaxScaler(feature_range = (0, 1))
df_train = df[:int(leng*Train_percent)]
df_test = df[int(leng*Train_percent):]

df_processed = df.iloc[:, 1:2].values  

df = scaler.fit_transform(df_processed)  
trainLength = int(len(df)*.8)
totalLength = len(df)
#x for the first sinus
data = np.empty((1,totalLength,2))

#the first sinus

#x for the second sinus - this reflects in a sinus with a different frequency

#the first * the second sinus 
data[0,:,1] = df.reshape(len(df))


def takeSlice(arr, fr, to, name):
    
    result = arr[:,fr:to,:]
    print(name + ": start at " + str(fr) + " - shape: " + str(result.shape))
    return result


#training data: y is one step ahead of x

x = takeSlice(data,0,trainLength,'x') #de 0 a 799
y = takeSlice(data,shift,shift+trainLength,'y') #de 7 a 806


#true data for forecasting:
xForecast = takeSlice(data,trainLength,-shift,'xForecast') #de 800 a 1192?
trueForecast = takeSlice(data,shift+trainLength,None,'trueForecast') #de 807 a 1199


#plotting

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(16,5))


ax.plot(data[0,:,1],color='b',linewidth=3)
fig.suptitle('these are the two features of the complete data',fontsize=20)
plt.show()
print('\n\n\n\n')

#figure separating training and forecasting
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(16,5))
ax.plot(range(trainLength),x[0,:,1],color='b',linewidth=1)
ax.plot(range(trainLength,totalLength-shift),xForecast[0,:,0],color='y',linewidth=4)
ax.plot(range(trainLength,totalLength-shift),xForecast[0,:,1],color='k',linewidth=1)
fig.suptitle('Training and test data put together: ', fontsize=20)
plt.show()

model = Sequential()
model.add(LSTM(100,return_sequences=True,input_shape=(None,2))) #input takes any steps, two features (var1 and var2)
model.add(Dropout(0.2))

model.add(LSTM(70,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(2,return_sequences=True)) #output keeps the steps and has two features
model.add(Dropout(0.2))
model.add(Lambda(lambda x: x*1.3))
#training the model

#this callback interrupts training when loss stops decreasing after 10 consecutive epochs. 
from keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss',min_delta=0.000000000001,patience=30) #this big patience is important

#different learning rates - train each indefinitely until the loss stops decreasing
#fount that the best rate is between 0.0001 and 0.00001
rates = [0.001]
for rate in rates:
    print('training with lr = ' + str(rate))
    model.compile(loss='mse', optimizer=Adam(lr=rate))
    model.fit(x,y,epochs=100,callbacks=[stop],verbose=1) #train indefinitely until loss stops decreasing
    print('\n\n\n\n\n')

#the model for predictions - copies the other model, but uses `return_sequences=False` and `stateful=True`
#the change is just to allow predicting step by step and using the predictions as new steps. 
newModel = Sequential()
newModel.add(LSTM(100,return_sequences=True,stateful=True,batch_input_shape=(1,None,2)))
newModel.add(Dropout(0.2))
newModel.add(LSTM(70,return_sequences=True,stateful=True))
newModel.add(Dropout(0.2))
newModel.add(LSTM(50,return_sequences=True,stateful=True))
newModel.add(Dropout(0.2))
newModel.add(LSTM(2,return_sequences=False,stateful=True))
newModel.add(Dropout(0.2))
newModel.add(Lambda(lambda x: x*1.3))

newModel.set_weights(model.get_weights())


#predicting from the predictions themselves (gets the training data as input to set states)
newModel.reset_states()

lastSteps = np.empty((1,totalLength-trainLength,2)) #includes a shift at the beginning to cover the gap 
lastSteps[:,:shift] = x[:,-shift:] #the initial shift steps are filled with x training data 
newModel.predict(x[:,:-shift,:]).reshape(1,1,2) #just to adjust states, predict with x without the last shift elements

rangeLen = totalLength-trainLength-shift
print('rangeLen: ', rangeLen)
for i in range(rangeLen):
    lastSteps[:,i+shift] = newModel.predict(lastSteps[:,i:i+1,:]).reshape(1,1,2)
print(lastSteps.shape)
forecastFromSelf = lastSteps[:,shift:,:]
print(forecastFromSelf.shape)


#predicting from test/future data:
newModel.reset_states()

newModel.predict(x) #just to set the states and get used to the sequence
newSteps = []
for i in range(xForecast.shape[1]):
    newSteps.append(newModel.predict(xForecast[:,i:i+1,:]))
forecastFromInput = np.asarray(newSteps).reshape(1,xForecast.shape[1],2)

print('trueForecast: ', trueForecast.shape)
print('forecastFromSelf: ', forecastFromSelf.shape)
print('forecastFromInput: ', forecastFromInput.shape)
print('\n\n\nblack line: true values')
print('gold line: predicted values')


#self forecast
fig,ax = plt.subplots(1,1,figsize=(16,5))
ax.plot(xForecast[0,:,0], linewidth=7,color='k') #this uses xForecast because it starts exactly where x ends
ax.plot(forecastFromSelf[0,:,0],color='y')
plt.suptitle("predicting feature 1 - self predictions")
plt.show()

fig,ax = plt.subplots(1,1,figsize=(16,5))
ax.plot(xForecast[0,:,1],linewidth=7,color='k') #this uses xForecast because it starts exactly where x ends
ax.plot(forecastFromSelf[0,:,1],color='y')
plt.suptitle("predicting feature 2 - self predictions")
plt.show()


#forecast from test/future data:
fig,ax = plt.subplots(1,1,figsize=(16,5))
ax.plot(trueForecast[0,:,0], linewidth=7,color='k')
ax.plot(forecastFromInput[0,:,0],color='y')
plt.suptitle("predicting feature 1 - predictions from true data")
