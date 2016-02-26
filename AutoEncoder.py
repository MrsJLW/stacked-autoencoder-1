import pickle
import numpy as np
import tensorflow as tf
from utils import *

np.random.seed(0)
tf.set_random_seed(0)


class DenoisingAutoEncoder(object):
    """ Denoising Autoencoder with an sklearn-like interface implemented using TensorFlow.                                                                                 
    adapted from https://jmetzen.github.io/2015-11-27/dae.html                                                                                     
    
    """
    def __init__(self, network_architecture,learning_rate=0.001, batch_size=100):
        
        self.sess = tf.InteractiveSession()
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input                                                                 
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self.x_noisy = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        #self.W = tf.Variable(tf.random_uniform((network_architecture["n_input"], network_architecture["n_hidden"])))
        self.W = tf.Variable(xavier_init(network_architecture["n_input"], network_architecture["n_hidden"]))
        self.b_encode = tf.Variable(tf.ones([network_architecture["n_hidden"]], dtype=tf.float32))
        b_decode = tf.Variable(tf.ones([network_architecture["n_input"]], dtype=tf.float32))
        
        #encode
        #activation function - softmax, softplus, or tanh?                 
        self.h = tf.nn.tanh(tf.add(tf.matmul(self.x_noisy, self.W), self.b_encode))
        
        #decode
        self.output = tf.nn.tanh(tf.add(tf.matmul(self.h, tf.transpose(self.W)),b_decode))
        
        # _ = tf.histogram_summary('weights', self.W1)
        # _ = tf.histogram_summary('biases_encode', self.b1_encode)
        # _ = tf.histogram_summary('biases_decode', b1_decode)
        # _ = tf.histogram_summary('hidden_units', self.h1)
        # _ = tf.histogram_summary('weights', self.W2)
        # _ = tf.histogram_summary('biases_encode', b2_encode)
        # _ = tf.histogram_summary('biases_decode', b2_decode)
        # _ = tf.histogram_summary('hidden_units', self.h2)

        with tf.name_scope("layer1") as scope:
            with tf.name_scope("loss") as scope:
                #loss function
                self.cost = tf.reduce_mean(tf.square(self.x - self.output))
                cost_summ = tf.scalar_summary("cost summary", self.cost)
        
            with tf.name_scope("train") as scope:
                #optimizer
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.merged = tf.merge_all_summaries()
        #self.writer = tf.train.SummaryWriter('%s/%s' % ("/tmp/mnist_logs", run_var), self.sess.graph_def)
        self.writer = tf.train.SummaryWriter("/tmp/mnist_logs", self.sess.graph_def)
        tf.initialize_all_variables().run()
                
        
        
    def log_stats(self, X, XN):
        result = self.sess.run(self.merged, feed_dict={self.x: X, self.x_noisy: XN})
        self.writer.add_summary(result)

            
    def partial_fit(self, X, XN):
        """Train model based on mini-batch of input data.                                
        Return cost of mini-batch.                                                       
        """
        self.sess.run(self.optimizer, feed_dict={self.x: X, self.x_noisy: XN})
        return self.sess.run(self.cost, feed_dict={self.x: X, self.x_noisy: XN})
        
                
    def reconstruct(self, X, XN):
        #encode            
        self.sess.run(self.h, feed_dict={self.x: X, self.x_noisy: XN})
        #decode
        return self.sess.run(self.output,feed_dict={self.x: X, self.x_noisy: XN}), self.sess.run(self.cost,feed_dict={self.x: X, self.x_noisy: XN})

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def train(network_architecture, dataset, validset, learning_rate=0.0001,
          batch_size=10, training_epochs=10, display_step=1, n_samples=1000, noise=1):
    
    print('Start training......')
    decayRate = 0.8
    dae = DenoisingAutoEncoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    # Training cycle                                                                     
    trainCost = []
    testCost = []
    print('train autoencoder')
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        # Loop over all batches                                                          
        for i in range(total_batch):
            batch_xs= dataset[i*batch_size: (i+1)*batch_size ]

            # Fit training using batch data                                              
            dae.learning_rate = dae.learning_rate * decayRate
            if(noise):
                cost = dae.partial_fit(batch_xs, removeNoise(batch_xs, 0.5))
            else:
                cost = dae.partial_fit(batch_xs, batch_xs)
            # Compute average loss                                                       
            avg_cost += cost / n_samples * batch_size
            
        # Display logs per epoch step --compare to validation cost                                                   
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
            trainCost.append("{:.9f}".format(avg_cost))
            x_reconstruct,testcost = dae.reconstruct(validset, validset)
            testCost.append(testcost)

    return dae, trainCost, testCost







######################################################################################
#####    Train 2 Stacked AutoEncoers. Save output to pickle file                 #####
######################################################################################


image_size = 28
num_labels = 10
pickle_file = '/media/caitlin/UbiComp2015/notMNIST/notMNIST_All.pickle'

print('getting data...')
train_dataset, train_labels,valid_dataset, valid_labels, test_dataset, test_labels = getnotMNISTData(image_size, num_labels, pickle_file)

network_architecture = dict(n_input=784,
                            n_hidden=100)


n_samples=40000
n_valid = 5000
x_sample = valid_dataset[0:n_valid]

#train first autoencoder
dae, trainCost1, testCost1 = train(network_architecture, train_dataset, x_sample, batch_size=100, training_epochs=8, learning_rate=5., n_samples=n_samples, noise=0)
print("train cost: ", trainCost1)

#reconstruct input
# x_reconstruct,testcost = dae.reconstruct(x_sample, x_sample)
# print("test cost: ", testcost)
# saveReconFig('reconstruction.png', x_sample, x_reconstruct, 5)

#plot training cost
plotTraining(trainCost1, testCost1, 'firstAutoEncoderTraining.png')

# output hidden Layer
W1 = dae.sess.run(dae.W)
print('w1 shape: ', W1.shape)
printWeights(W1, 'hiddenLayer1.png')

#fix weights
b_encode = dae.sess.run(dae.b_encode)
b1 = np.tile(b_encode,(n_samples,1))
train_dataset2 = np.tanh(np.add(np.dot(train_dataset[0:n_samples], W1), b1))
bv = np.tile(b_encode,(n_valid,1))
valid_dataset2 = np.tanh(np.add(np.dot(valid_dataset[0:n_valid], W1), bv))


network_architecture = dict(n_input=100, # 1st layer encoder neurons
                            n_hidden=64) # MNIST data input (img shape: 28*28)

#train second autoencoder
dae, trainCost2, testCost2 = train(network_architecture, train_dataset2, valid_dataset2, batch_size=100, training_epochs=1, learning_rate=5., n_samples=n_samples, noise=0)
print("train cost: ", trainCost2)

# output hidden Layer
W2 = dae.sess.run(dae.W)
printWeights(W2, 'hiddenLayer2.png')

#plot training cost
plotTraining(trainCost2, testCost2, 'secondAutoEncoderTraining.png')

#fix weights, train second autoencoder
b_encode = dae.sess.run(dae.b_encode)
b2 = np.tile(b_encode,(n_samples,1))

pickle_file = 'Weights.pickle'
print('w1: ', W1.shape)
print('w2: ', W1.shape)
try:
   f = open(pickle_file, 'wb')
   save = {
       'W1Layer1': W1,
       'W1Layer2' : W2,
       'b1Layer1' : b1,
       'b1Layer2': b2,
    }
   pickle.dump(save, f, protocol = 2)
   f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
