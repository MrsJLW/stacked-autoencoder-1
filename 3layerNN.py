import pickle
import numpy as np
import tensorflow as tf
from utils import *
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


np.random.seed(0)
tf.set_random_seed(0)


class NN(object):
    """ Denoising Autoencoder with an sklearn-like interface implemented using TensorFlow.                                                                                 
    adapted from https://jmetzen.github.io/2015-11-27/vae.html                                                                                     
    
    """
    def __init__(self, network_architecture,learning_rate=0.001, batch_size=100):
        
        self.sess = tf.InteractiveSession()
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input                                                                 
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        #labels
        self.y = tf.placeholder(tf.float32, [None, network_architecture["n_classes"]])
        # W1 = tf.Variable(tf.random_uniform((network_architecture["n_input"], network_architecture["n_hidden1"])))
        # W2 = tf.Variable(tf.random_uniform((network_architecture["n_hidden1"], network_architecture["n_hidden2"])))
        # W3 = tf.Variable(tf.random_uniform((network_architecture["n_hidden2"], network_architecture["n_hidden3"])))
        # W4 = tf.Variable(tf.random_uniform((network_architecture["n_hidden3"], network_architecture["n_classes"])))
        W1 = tf.Variable(xavier_init(network_architecture["n_input"], network_architecture["n_hidden1"]))
        W2 = tf.Variable(xavier_init(network_architecture["n_hidden1"], network_architecture["n_hidden2"]))
        W3 = tf.Variable(xavier_init(network_architecture["n_hidden2"], network_architecture["n_hidden3"]))
        W4 = tf.Variable(xavier_init(network_architecture["n_hidden3"], network_architecture["n_classes"]))
        b1 = tf.Variable(tf.ones([network_architecture["n_hidden1"]], dtype=tf.float32))
        b2 = tf.Variable(tf.ones([network_architecture["n_hidden2"]], dtype=tf.float32))
        b3 = tf.Variable(tf.ones([network_architecture["n_hidden3"]], dtype=tf.float32))
        b4 = tf.Variable(tf.ones([network_architecture["n_classes"]], dtype=tf.float32))

        #activation function - softmax, softplus, or tanh?
        #hidden layer
        self.h1 = tf.nn.softmax(tf.add(tf.matmul(self.x, W1), b1))
        self.h2 = tf.nn.softmax(tf.add(tf.matmul(self.h1, W2), b2))
        self.h3 = tf.nn.softmax(tf.add(tf.matmul(self.h2, W3), b3))

        #prediction
        self.output = tf.nn.softmax(tf.add(tf.matmul(self.h3, W4),b4))
        
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
            
        
            
def train_nn(network_architecture, learning_rate=0.0001,
          batch_size=10, training_epochs=10, n_samples=1000):
    
    print('Start training......')
    decayRate = 0.1
    nn = NN(network_architecture, learning_rate=learning_rate, batch_size=batch_size)
    # Training cycle                                                                     
    trainCost = []
    testCost = []
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

        # Test trained model
        tempTrainAcc = nn.accuracy.eval({nn.x: train_dataset[0:n_samples], nn.y: train_labels[0:n_samples]})
        print('training:', tempTrainAcc)
        trainCost.append(tempTrainAcc)
        tempTestAcc = nn.accuracy.eval({nn.x: valid_dataset[0:100], nn.y: valid_labels[0:100]})
        print('testing:',tempTestAcc)
        testCost.append(tempTestAcc)
    return nn, trainCost

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)
############ helpers #######################################

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]                               
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

#get notMNIST data
#def getnotMNISTData(train_dataset, train_labels, valid_dataset, valid_labels):
pickle_file = 'notMNIST_All.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  test_d = save['test_dataset']
  test_l = save['test_labels']
  del save  # hint to help gc free up memory

  test_dataset = test_d[:4999]
  test_labels = test_l[:4999]
  valid_dataset = test_d[5000:,:,:]
  valid_labels = test_l[5000:]

  print ('Training set', train_dataset.shape, train_labels.shape)                     
  print ('Validation set', valid_dataset.shape, valid_labels.shape)                  
  print ('Test set', test_dataset.shape, test_labels.shape)                           

image_size = 28
num_labels = 10

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print ('Training set', train_dataset.shape, train_labels.shape)
print ('Validation set', valid_dataset.shape, valid_labels.shape)
print ('Test set', test_dataset.shape, test_labels.shape)



network_architecture = dict(n_hidden1=500,
                            n_hidden2=100,
                            n_hidden3=64,
                            n_input=784,
                            n_classes=10)

x_sample = test_dataset[0:100]
y_sample = test_labels[0:100]

nn, trainCost, testCost = train_nn(network_architecture, batch_size=100, training_epochs=100, learning_rate=10., n_samples=10000)
x_reconstruct,testcost = nn.predict(x_sample, y_sample)
print("test cost: ", testcost)

plt.clf()
plt.plot(trainCost, label='training Cost')
plt.plot(testCost, label='Validation Cost')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('3layerNN_accuray.png')
# Test trained model
# print(nn.auc.eval({nn.x: x_sample, nn.y_: y_sample}))
   
#aveReconFig('All_letters_no_noise_50000train_1000h.png', x_sample, x_reconstruct, 5)

