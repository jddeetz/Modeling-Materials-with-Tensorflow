import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

#Load the data
data_store = pickle.load(open("data.pkl",'rb'))  
X_train,y_train,labels_train,X_valid,y_valid,labels_valid=data_store[0].T,data_store[1],data_store[2],data_store[3].T,data_store[4],data_store[5]

#Reshape y_train and y_test and keep from being annoying
y_train=y_train.reshape(y_train.shape[0],1).T
y_valid=y_valid.reshape(y_valid.shape[0],1).T

#Define Architecture of Network
num_nodes=[np.shape(X_train)[0],80,40,20,8,1]

#Create placeholders for X and Y values, so that we can later pass training data to them
def create_placeholders(n_x,n_y):
    """Creates the placeholders for the tensorflow session
    n_x and n_y are scalars. n_x is the size of the input vector.
    n_y is the number of outputs.
    
    Returns X, a placeholder for the data input of shape [n_x,None]
    Returns Y, a placeholder for the outputs of shape [n_y,None]
    Use None here because it allows flexibility on the number of examples 
    used for the placeholders
    """
    X = tf.placeholder(tf.float32, shape=(n_x,None), name="X")
    Y = tf.placeholder(tf.float32, shape=(n_y,None), name="Y")
    
    return X, Y
    
#Function for initializing the parameters
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow.
    Returns a dictionary of tensors containing W and b parameters
    """
    W1 = tf.get_variable("W1", [num_nodes[1],num_nodes[0]], initializer = tf.contrib.layers.xavier_initializer(seed = 100))
    b1 = tf.get_variable("b1", [num_nodes[1],1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [num_nodes[2],num_nodes[1]], initializer = tf.contrib.layers.xavier_initializer(seed = 2))
    b2 = tf.get_variable("b2", [num_nodes[2],1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [num_nodes[3],num_nodes[2]], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
    b3 = tf.get_variable("b3", [num_nodes[3],1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [num_nodes[4],num_nodes[3]], initializer = tf.contrib.layers.xavier_initializer(seed = 4))
    b4 = tf.get_variable("b4", [num_nodes[4],1], initializer = tf.zeros_initializer())
    W5 = tf.get_variable("W5", [num_nodes[5],num_nodes[4]], initializer = tf.contrib.layers.xavier_initializer(seed = 5))
    b5 = tf.get_variable("b5", [num_nodes[5],1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5}
    
    return parameters

#Forward propagation function
def forward_propagation(X, parameters):
    """
    Implements forward propagation for the model LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> RELU ... ->LINEAR
    
    Takes X, the input dataset placeholder, of shape (inputsize, m)
    Takes parameters: a dictionary of the W and b parameters
    
    Returns: Z3 --- the output of the last unit
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    
    Z1 = tf.matmul(W1, X)+b1                               
    A1 = tf.nn.relu(Z1)                                    
    Z2 = tf.matmul(W2, A1)+b2                              
    A2 = tf.nn.relu(Z2)                                    
    Z3 = tf.matmul(W3, A2)+b3                              
    A3 = tf.nn.relu(Z3)
    Z4 = tf.matmul(W4, A3)+b4                              
    A4 = tf.nn.relu(Z4)
    Z5 = tf.matmul(W5, A4)+b5                              
    A5 = tf.nn.relu(Z5)
    
    return A5

#Compute the cost function
def compute_cost(A5, Y):
    """
    Computes mean squared error cost function
    
    Takes A3 -- the output of the network, the activations of the final layer
    Takes Y -- the set of training examples
    
    Returns: cost -- a summary of the error versus the training set.
    """
    #cost = tf.reduce_mean(tf.pow(A5 - Y,2))
    cost = tf.reduce_mean(tf.pow(A5 - Y,2))
    return cost
    
#Bring it all together
def model(X_train,y_train,X_valid,y_valid, learning_rate=0.0001,num_epochs=2000,print_cost = True):
    """
    Implements a five-layer tensorflow neural network linear->RELU->linear->RELU->linear->RELU
    
    Arguments:
    X_train,Y_train -- training set
    X_test,Y_test -- test set
    learning_rate -- learning rate for gradient descent optimization
    num_epochs -- number of passes through the training set, iterations of gradient descent
    print_cost -- True to print the cost every 100 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph() #in order to rerun the model without overwriting the tf variables
    #Define some variables
    (n_x,m) = X_train.shape
    print('There are',m,'training examples')
    print('There are',n_x,'inputs')
    n_y = y_train.shape[0]
    print('There are',n_y,'outputs')
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    
    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    A5 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(A5, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch 
            # Run the session to execute the "optimizer" and the "cost", the feedict is X_train,y_train for (X,Y).
            _ , epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: y_train})
            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate accuracy on the train set
        accuracy = tf.reduce_mean(tf.pow(A5 - Y,2))
        print ("Training Set Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
        
        # Calculate accuracy on the validation set
        (n_x,m) = X_valid.shape #shape 33 108
        n_y = y_valid.shape[0] #shape 108
        X, Y = create_placeholders(n_x, n_y)
        A5 = forward_propagation(X, parameters)
        accuracy = tf.reduce_mean(tf.pow(A5 - Y,2))
        
        print ("Validation Set Accuracy:", accuracy.eval({X: X_valid, Y: y_valid}))
        
        return parameters

#Optimize the parameters
parameters = model(X_train, y_train, X_valid, y_valid)

#Predict densities of validation set
def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])
    W5 = tf.convert_to_tensor(parameters["W5"])
    b5 = tf.convert_to_tensor(parameters["b5"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W4": W4,
              "b4": b4,
              "W5": W5,
              "b5": b5}
    
    x = tf.placeholder("float", [33, 1])
    
    a5 = forward_propagation(x, params)
    
    with tf.Session() as sess:
        prediction = sess.run(a5, feed_dict = {x: X})
        
    return prediction

#Execute prediction of densities of validation set
sum_error=list()
for i in range(X_valid.shape[1]):
    fingerprint=X_valid[:,i]
    fingerprint=fingerprint.reshape(fingerprint.shape[0],1)
    density_prediction = predict(fingerprint, parameters)
    print('Material:',labels_valid[i],', Actual density: ', y_valid[0][i],', Predicted density: ',density_prediction)
    sum_error.append(abs(float((y_valid[0][i]-density_prediction)/y_valid[0][i])))

ave_error=sum(sum_error)/X_valid.shape[1]*100
print('The average error of the predicted densities is',ave_error,'%')