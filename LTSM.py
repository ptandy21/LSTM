
import pandas as pd 
import os
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
from sklearn import preprocessing


%load_ext tensorboard.notebook
df = pd.read_excel(r"C:\Users\ptandy\Desktop\ML_TEST_DATA.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
idf = df.set_index(['Date'])
sess=tf.Session()
SEQ_LEN = 5
Future = 2
from collections import deque
import random
import time
EPOCHS = 8 
BATCH_SIZE = 20
NAME = f"{SEQ_LEN}-SEQ-{Future}-PRED-{int(time.time())}"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

from sklearn import preprocessing 
def preprocessed_df(df):
    df = df.drop('future',1)
    for col in df.columns:
        if col != 'target':
            df['M. Dollars'] = df['M. Dollars'].pct_change()
            df.dropna(inplace = True)
            df['M. Dollars'] = preprocessing.scale(df['M. Dollars'].values)
            
    df.dropna(inplace = True)
    
    seq_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            seq_data.append([np.array(prev_days), i[-1]])
    random.shuffle(seq_data)
    
    buys =[]
    sells = []
    
    for seq, target in seq_data:
        if target == 0:
            sells.append([seq,target])
        elif target == 1:
            buys.append([seq,target])
            
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    seq_data = buys + sells 
    random.shuffle(seq_data)
    X = []
    y = []
    
    for seq, target in seq_data:
        X.append(seq)
        y.append(target)
        
    return np.array(X), y
    
    
    
    
idf['future'] = idf['M. Dollars'].shift(-Future)
idf['target'] = list(map(classify, idf['M. Dollars'], idf['future']))


times = sorted(idf.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_df = idf[(idf.index >= last_5pct)]
idf = idf[(idf.index < last_5pct)]

train_x, train_y = preprocessed_df(idf)
validation_x, validation_y = preprocessed_df(validation_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont Buys: {train_y.count(0)}, buys {train_y.count(1)}")
print(f"Validation Dont buys: {validation_y.count(0)}, buys:{validation_y.count(1)}")


model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("C:/Users/ptandy/Desktop/models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("C:/Users/ptandy/Desktop/models/{}".format(NAME))

file_writer = tf.summary.FileWriter('C:/Users/ptandy/Desktop/models',sess.graph)