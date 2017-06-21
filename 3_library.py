import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers


def prepare_data():
    y = np.genfromtxt('train_targets.csv', delimiter=',', dtype=int)
    X = np.genfromtxt('train_data.csv', delimiter=',')
    test = np.genfromtxt('test_data.csv', delimiter=',')
    return X, y, test


def output_res(y_hat):
    with open('test_predictions_library.csv', 'w') as file:
        for i in range(y_hat.size):
            file.write(str(y_hat[i]) + "\r\n")


X, y, test = prepare_data()
l2_reg = regularizers.l2(5e-5)


model = Sequential()
model.add(Dense(units=512, input_dim=400, init='glorot_normal', kernel_regularizer=l2_reg))
model.add(Activation('relu'))
model.add(Dense(units=512, init='glorot_normal', kernel_regularizer=l2_reg))
model.add(Activation('relu'))
model.add(Dense(units=10, init='glorot_normal', kernel_regularizer=l2_reg))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.05, momentum=0.9, nesterov=True, decay=.01))

model.fit(X, y, epochs=30, batch_size=20, verbose=1)

y_hat = model.predict(test)
y_hat = np.argmax(y_hat, axis=1)

output_res(y_hat)
