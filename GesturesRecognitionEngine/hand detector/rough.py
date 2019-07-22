import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd

s = tf.InteractiveSession()

#def load_dataset(flatten=False):
#    with open('coord.txt') as f:
#        tl,tr,bl,br = [str(x) for x in next(f).split()]
#        array = [[int(x) for x in line.split()] for line in f]
#    cla=np.zeros([9000,9])
#    all = np.zeros([9000,4])
#    l_r,u_r=0,1000
#    for po in range(9):
#        coord=array[po]
#        all[l_r:u_r,:]=coord
#        cla[l_r:u_r,po]=1
#        l_r,u_r=l_r+1000,u_r+1000
#    X_train=all
#    y_train=cla
#    X_test=np.array([[3,29,5,22],[30,57,36,64],[72,98,60,85]],np.float64)
#    y_test=np.array([[1,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,1]],np.float64)
#    return X_train, y_train, X_test,y_test

heart = pd.read_csv('centers.csv')
X_tr=[]
Y_tr=[]
X = heart.iloc[0:9, 0:2]
Y = heart.iloc[0:9, 2]
Y=pd.Series.tolist(Y)
for i in range(9):
    X_base=pd.Series.tolist(X.iloc[i,:])
    xl_lim=int(X_base[0])-30
    xu_lim=int(X_base[0])+30
    yl_lim=int(X_base[1])-30
    yu_lim=int(X_base[1])+30
    for xlim in range(xl_lim,xu_lim):
        for ylim in range(yl_lim,yu_lim):
            X_tr.append([xlim,ylim])
            Y_tr.append(int(Y[i]))
            
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tr, Y_tr, test_size=0.3, random_state=0)
#X_train, y_train, X_test, y_test = load_dataset()
#XX = np.array([[3,29,5,22]],np.float64)
#yy = np.array([1,0,0,0,0,0,0,0,0])

## Changing dimension of input images from N*28*28 to  N*784
#X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
#X_test = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))

#print('Train dimension:');print(X_train.shape)
#print('Test dimension:');print(X_test.shape)

## Defining various initialization parameters for 784-512-256-10 MLP model
num_classes = 9 #y_train.shape[1]
num_features = 2 #X_train.shape[1]
num_output = 9 #y_train.shape[1]
num_layers_0 = 18
#num_layers_1 = 250
#num_layers_2 = 25
starter_learning_rate = 0.001
regularizer_rate = 0.1

# Placeholders for the input data
input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')

## for dropout layer
keep_prob = tf.placeholder(tf.float32)

## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
bias_0 = tf.Variable(tf.random_normal([num_layers_0]))

#weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_layers_1], stddev=(1/tf.sqrt(float(num_layers_0)))))
#bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
#
#weights_2 = tf.Variable(tf.random_normal([num_layers_1,num_layers_2], stddev=(1/tf.sqrt(float(num_layers_1)))))
#bias_2 = tf.Variable(tf.random_normal([num_layers_2]))

weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_output], stddev=(1/tf.sqrt(float(num_layers_0)))))
bias_1 = tf.Variable(tf.random_normal([num_output]))

## Initializing weigths and biases
hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)

#hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0,weights_1)+bias_1)
#hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
#
#hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1_1,weights_2)+bias_2)
#hidden_output_2_2 = tf.nn.dropout(hidden_output_2, keep_prob)

predicted_y = tf.sigmoid(tf.matmul(hidden_output_0_0,weights_1) + bias_1)

## Defining the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y,labels=input_y)) \
        + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))

## Variable learning rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)

## Adam optimzer for finding the right weight
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,weights_2,bias_0,bias_1,bias_2])
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,bias_0,bias_1])
    
## Metrics definition
correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Training parameters
batch_size = 3
epochs=5
dropout_prob = 0.5

training_accuracy = []
training_loss = []
testing_accuracy = []

s.run(tf.global_variables_initializer())
for epoch in range(epochs):    
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)
    for index in range(0,X_train.shape[0],batch_size):
        s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]],
                          input_y: y_train[arr[index:index+batch_size]],
                        keep_prob:dropout_prob})
#        print(index)
    training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, 
                                                         input_y: y_train,keep_prob:1}))
    training_loss.append(s.run(loss, {input_X: X_train, 
                                      input_y: y_train,keep_prob:1}))
    
    # Evaluation of model
    testing_accuracy.append(accuracy_score(y_test.argmax(1), 
                            s.run(predicted_y, {input_X: X_test,keep_prob:1}).argmax(1)))
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,
                                                                    training_loss[epoch],
                                                                    training_accuracy[epoch],
                                                                   testing_accuracy[epoch]))

    
    ## Plotting chart of training and testing accuracy as a function of iterations
iterations = list(range(epochs))
plt.plot(iterations, training_accuracy, label='Train')
plt.plot(iterations, testing_accuracy, label='Test')
plt.ylabel('Accuracy')
plt.xlabel('iterations')
plt.show()
print("Train Accuracy: {0:.2f}".format(training_accuracy[-1]))
print("Test Accuracy:{0:.2f}".format(testing_accuracy[-1]))