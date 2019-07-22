import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd

heart = pd.read_csv('coordinates.csv')
X = heart.iloc[:, 0:4]
#print(X)
Y = heart.iloc[:, 4]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

num_of_classes=9
model = Sequential()
model.add(Dense(20,input_shape=(4,1),activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(num_of_classes, activation='softmax'))
sgd = optimizers.SGD(lr=1e-2)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

for i in range(90):
    model.fit(X_train[:,i], Y_train[i], validation_data=(X_test[:,i], Y_test[i]), epochs=10, batch_size=2)
    scores = model.evaluate(X_test, Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))