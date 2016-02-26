import pickle
import numpy as np
import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


np.random.seed(0)
tf.set_random_seed(0)


class NN(object):
    """ Denoising Autoencoder with an sklearn-like interface implemented using TensorFlow.                                                                                 
    adapted from https://jmetzen.github.io/2015-11-27/vae.html                                                                                     
    
    """
    def __init__(self, network_architecture, network_layers, learning_rate=0.001, batch_size=100):
        
        self.sess = tf.InteractiveSession()
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input                                                                 
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        #labels
        self.y = tf.placeholder(tf.float32, [None, network_architecture["n_classes"]])


        self.W1 = tf.Variable(network_layers['W1']) #from autoencoder1
        self.W2 = tf.Variable(network_layers['W2']) #from autoencoder2
        #NN weights
        self.W3 = tf.Variable(tf.random_uniform((network_layers['W2'].shape[1], network_architecture["n_classes"])))
        b1 = tf.Variable(network_layers['b1']) #from autoencoder1
        b2 = tf.Variable(network_layers['b2']) #from autoencoder2
        #NN bias
        b3 = tf.Variable(tf.ones([network_architecture["n_classes"]], dtype=tf.float32))

        print('W3', self.W3.get_shape())
        #activation function - softmax, softplus, or tanh?
        #hidden layer
        self.h1 = tf.nn.tanh(tf.add(tf.matmul(self.x, self.W1), b1))
        self.h2 = tf.nn.tanh(tf.add(tf.matmul(self.h1, self.W2), b2))
        print('h2', self.h2.get_shape())
        #prediction
        self.output = tf.nn.softmax(tf.add(tf.matmul(self.h2, self.W3),b3))
        
        # _ = tf.histogram_summary('weights', W)
        # _ = tf.histogram_summary('biases_encode', b_encode)
        # _ = tf.histogram_summary('biases_decode', b_decode)
        # _ = tf.histogram_summary('hidden_units', self.h)

        with tf.name_scope("loss") as scope:
            #loss function mean squared error
            self.cost = tf.reduce_mean(tf.square(self.y - self.output))
            # cost_summ = tf.scalar_summary("cost summary", self.cost)
        
        with tf.name_scope("train") as scope:
            #optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        # Merge all the summaries and write them out to /tmp/mnist_logs
        self.merged = tf.merge_all_summaries()
        #self.writer = tf.train.SummaryWriter('%s/%s' % ("/tmp/mnist_logs", run_var), self.sess.graph_def)
        self.writer = tf.train.SummaryWriter("/tmp/mnist_logs", self.sess.graph_def)

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # self.auc = roc_auc_score(self.sess.run(self.y), self.sess.run(self.output))

        tf.initialize_all_variables().run()

        
    def log_stats(self, X):
        result = self.sess.run(self.merged, feed_dict={self.x: X, self.y: Y})
        self.writer.add_summary(result)

            
    def train(self, X, Y):
        self.sess.run(self.optimizer, feed_dict={self.x: X, self.y: Y})
        return self.sess.run(self.cost, feed_dict={self.x: X, self.y: Y})
                
    def predict(self, X, Y):
        return self.sess.run(self.output,feed_dict={self.x: X, self.y: Y}),\
               self.sess.run(self.cost,feed_dict={self.x: X, self.y: Y})

    def getWeights(self):
        return self.sess.run(self.W1), self.sess.run(self.W2), self.sess.run(self.W3)
            
        
            
def train_nn(network_architecture, network_layers, learning_rate=0.0001,
          batch_size=10, training_epochs=10, n_samples=1000):
    
    print('Start training......')
    decayRate = 0.99
    nn = NN(network_architecture, network_layers, learning_rate=learning_rate, batch_size=batch_size)
    # Training cycle                                                                     
    trainCost = []
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        # Loop over all batches                                                          
        for i in range(total_batch):
            batch_xs= train_dataset[i*batch_size: (i+1)*batch_size]
            batch_ys= train_labels[i*batch_size: (i+1)*batch_size]
            # Fit training using batch data                                              
            nn.learning_rate = nn.learning_rate * decayRate
            cost = nn.train(batch_xs, batch_ys)
            # Compute average loss                                                       
            avg_cost += cost / n_samples * batch_size
            # nn.log_stats(batch_xs)
            
        # Display logs per epoch step                                                    
        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost))
            trainCost.append("{:.9f}".format(avg_cost))

        # Test trained model
        print('training:', nn.accuracy.eval({nn.x: train_dataset[0:n_samples], nn.y: train_labels[0:n_samples]}))
        print('testing:', nn.accuracy.eval({nn.x: valid_dataset[0:100], nn.y: valid_labels[0:100]}))

    w1, w2, w3 = nn.getWeights()

    printWeights(w1, 'W1.png')
    printWeights(w2, 'W2.png')
    printWeights(w3, 'W3.png')

    return nn, trainCost

############ helpers #######################################


image_size = 28
num_labels = 10
pickle_file = '/media/caitlin/UbiComp2015/notMNIST/notMNIST_All.pickle'

print('getting data...')
train_dataset, train_labels,valid_dataset, valid_labels, test_dataset, test_labels = getnotMNISTData(image_size, num_labels, pickle_file)

print ('Training set', train_dataset.shape, train_labels.shape)
print ('Validation set', valid_dataset.shape, valid_labels.shape)
print ('Test set', test_dataset.shape, test_labels.shape)

network_architecture = dict(n_hidden=500, # 1st layer encoder neurons
                            n_input=784, # MNIST data input (img shape: 28*28)
                            n_classes=10)

pickle_file = '/media/caitlin/UbiComp2015/notMNIST/SecondLayerWeights.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  network_layers = dict(W1 = save['W1Layer1'],
                        b1 = save['b1Layer1'],
                        W2 = save['W1Layer2'],
                        b2 = save['b1Layer2'])
  del save  # hint to help gc free up memory


print('W2 shape', network_layers['W2'].shape[1])

x_sample = test_dataset[0:100]
y_sample = test_labels[0:100]

nn, trainCost4 = train_nn(network_architecture, network_layers, batch_size=100, training_epochs=10, learning_rate=1.,n_samples=10000)
x_reconstruct,testcost = nn.predict(x_sample, y_sample)
print("test cost: ", testcost)
# Test trained model
# print(nn.auc.eval({nn.x: x_sample, nn.y_: y_sample}))
   
#saveReconFig('All_letters_no_noise_50000train_1000h.png', x_sample, x_reconstruct, 5)

