import pickle
import numpy as np
import tensorflow as tf
from utils import *
import PIL
import matplotlib.pyplot as plt
import DrawHiddenLayers
import classifier

np.random.seed(0)
tf.set_random_seed(0)


class DenoisingAutoEncoder(object):
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
        self.x_noisy = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        self.W1 = tf.Variable(xavier_init(network_architecture["n_input"], network_architecture["n_hidden1"]))
        self.b1_encode = tf.Variable(tf.zeros([network_architecture["n_hidden1"]], dtype=tf.float32))
        b1_decode = tf.Variable(tf.zeros([network_architecture["n_input"]], dtype=tf.float32))
        #encode
        #activation function - softmax, softplus, or tanh?                 
        self.h1 = tf.nn.tanh(tf.add(tf.matmul(self.x_noisy, self.W1), self.b1_encode))

        #decode
        self.output1 = tf.nn.tanh(tf.add(tf.matmul(self.h1, tf.transpose(self.W1)),b1_decode))

        with tf.name_scope("layer1") as scope:
            with tf.name_scope("loss") as scope:
                #loss function
                self.cost1 = tf.reduce_mean(tf.square(self.x - self.output1))
                cost_summ = tf.scalar_summary("cost summary", self.cost1)
        
            with tf.name_scope("train") as scope:
                #optimizer
                self.optimizer1 = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost1)

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
        self.sess.run(self.optimizer1, feed_dict={self.x: X, self.x_noisy: XN})
        return self.sess.run(self.cost1, feed_dict={self.x: X, self.x_noisy: XN})
        
                
    def reconstruct(self, XBig, XNBig, n_samples):
        total_batch = int(n_samples / self.batch_size)
        avg_cost = 0;
        # Loop over all batches
        for i in range(total_batch):
            batch_xs= XBig[i*self.batch_size: (i+1)*self.batch_size ]
            batch_xsn = XNBig[i*self.batch_size: (i+1)*self.batch_size ]
            # Fit training using batch data
            self.sess.run(self.h1, feed_dict={self.x: batch_xs, self.x_noisy: batch_xsn})
            ReconstructImage = self.sess.run(self.output1,feed_dict={self.x: batch_xs, self.x_noisy: batch_xsn})
            tempCost = self.sess.run(self.cost1,feed_dict={self.x: batch_xs, self.x_noisy: batch_xsn})
            # Compute average loss
            avg_cost += tempCost
        #decode
        return ReconstructImage,avg_cost/total_batch
        
            
def trainFirstLayer(train_dataset, network_architecture, learning_rate=0.0001,
          batch_size=10, training_epochs=10, display_step=1, n_samples=1000, noise=1):
    
    print('Start training......')
    decayRate = 0.99
    vae = DenoisingAutoEncoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    # Training cycle                                                                     
    trainCost = []
    testCost = []
    print('train first autoencoder')
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        # Loop over all batches                                                          
        for i in range(total_batch):
            batch_xs= train_dataset[i*batch_size: (i+1)*batch_size ]

            # Fit training using batch data                                              
            vae.learning_rate = vae.learning_rate * decayRate
            if(noise):
                cost = vae.partial_fit(batch_xs, removeNoise(batch_xs, 0.5))
            else:
                cost = vae.partial_fit(batch_xs, batch_xs)
            # Compute average loss                                                       
            avg_cost += cost / n_samples * batch_size
            
        # Display logs per epoch step                                                    
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost))
            trainCost.append("{:.9f}".format(avg_cost))
            x_sample = test_dataset[0:1000]
            x_reconstruct,testcost = vae.reconstruct(x_sample, x_sample,1000)
            testCost.append(testcost)
            print("test cost: ", testcost)
    print("multiplying...")
    
    W1 = vae.sess.run(vae.W1)
    # output hidden Layer
    image = PIL.Image.fromarray(DrawHiddenLayers.tile_raster_images(X=W1.T,
                                                                    img_shape=(28, 28), tile_shape=(10, 10),
                                                                    tile_spacing=(1, 1)))
    image.save('hiddenLayer1.png')   
    #
    # x_reconstruct,testcost = vae.reconstruct(x_sample, x_sample)
    # print("test cost: ", testcost)
    # saveReconFig('All_layers_2layer.png', x_sample, x_reconstruct, 5)

    b1_encode = vae.sess.run(vae.b1_encode)
    b1 = np.tile(b1_encode,(n_samples,1))
    second_dataset = np.tanh(np.add(np.dot(train_dataset[0:n_samples], W1), b1))
    return vae, second_dataset, W1, b1_encode, trainCost, testCost


def trainSecondLayer(second_dataset, network_architecture, learning_rate=0.0001,
          batch_size=10, training_epochs=10, display_step=1, n_samples=1000, noise=1):
    vae = DenoisingAutoEncoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)

    # Training cycle                                                                     
    trainCost = []
    decayRate = 0.99
    print('train second autoencoder')
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        # Loop over all batches                                                          
        for i in range(total_batch):
            batch_xs= second_dataset[i*batch_size: (i+1)*batch_size ]

            # Fit training using batch data                                              
            vae.learning_rate = vae.learning_rate * decayRate
            if(noise):
                cost = vae.partial_fit(batch_xs, removeNoise(batch_xs, 0.5))
            else:
                cost = vae.partial_fit(batch_xs, batch_xs)
            # Compute average loss                                                       
            avg_cost += cost / n_samples * batch_size
            
        # Display logs per epoch step                                                    
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost))
            trainCost.append("{:.9f}".format(avg_cost))
    
    W1 = vae.sess.run(vae.W1)
    # output hidden Layer
    image = PIL.Image.fromarray(DrawHiddenLayers.tile_raster_images(X=W1.T,
                                                                    img_shape=(28, 28), tile_shape=(10, 10),
                                                                    tile_spacing=(1, 1)))
    image.save('hiddenLayer2.png')   

    #x_reconstruct,testcost = vae.reconstruct(x_sample, x_sample)                           
    #print("test cost: ", testcost)
    #saveReconFig('All_layers_2layer.png', x_sample, x_reconstruct, 5)

    b1_encode = vae.sess.run(vae.b1_encode)
    b1 = np.tile(b1_encode,(n_samples,1))
    third_dataset = np.tanh(np.add(np.dot(second_dataset[0:n_samples], W1), b1))
    return third_dataset,  W1, b1_encode, trainCost

def trainClassifer (third_dataset, learning_rate=0.0001, decay_rate=0.99,
          batch_size=10, training_epochs=10, display_step=1, n_samples=1000, noise=1):
    cl = classifier.Classifier(learning_rate=learning_rate)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = third_dataset[i*batch_size: (i+1)*batch_size], train_labels[i*batch_size: (i+1)*batch_size]
            cl.train_step.run({cl.x: batch_xs, cl.y_: batch_ys})
            # Display logs per epoch step
        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "cross_entropy=", "{:.9f}".format(cl.cross_entropy.eval({cl.x: third_dataset, cl.y_: train_labels[0:n_samples]})))
        cl.learning_rate *= decay_rate

    # Test trained model
    print(cl.accuracy.eval({cl.x: third_dataset, cl.y_: train_labels[0:n_samples]}))


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow                                                                                 
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def removeNoise(training, prob):
    noisy_training = training.copy()
    for n in np.nditer(noisy_training, op_flags=['readwrite']):
        if np.random.random() < prob:
            n[...] = 0
    return noisy_training

def addNoise(training, prob):
    noisy_training = training.copy()
    for n in np.nditer(noisy_training, op_flags=['readwrite']):
        if np.random.random() < prob:
            n[...] = np.random.random()
    return noisy_training



############ helpers #######################################

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]                               
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def saveReconFig(title, x_sample, x_reconstruct, n):
  plt.figure(figsize=(8, 15))
  for i in range(n):
    plt.subplot(n, 2, 2*i + 1)
    plt.imshow(x_sample[i+n].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(n, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i+n].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Reconstruction")
    plt.colorbar()
  plt.savefig(title)





#get notMNIST data
#getnotMNISTData()
pickle_file = 'notMNIST_All.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  test_d = save['test_dataset']
  test_l = save['test_labels']
  del save  # hint to help gc free up memory

  test_dataset = test_d[:5000,:,:]
  test_labels = test_l[:5000]
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


network_architecture = dict(n_hidden1=784, # 1st layer encoder neurons
                            n_hidden2=784,
                            n_input=784) # MNIST data input (img shape: 28*28) 
                            

# vaeFirst, secondDataset, W1, b1_encode, trainCost,testCost = trainFirstLayer(train_dataset, network_architecture, batch_size=10, training_epochs=100, learning_rate=5.,
#                                 n_samples=5000, noise=0)
# x_sample = test_dataset[0:10]
# x_reconstruct,testcost = vaeFirst.reconstruct(x_sample, x_sample,10)
# print("test cost: ", testcost)
# saveReconFig('All_layers_1layer.png', x_sample, x_reconstruct, 5)
#
# # plot training cost and testing cost
# print(len(trainCost))
# print(len(testCost))
# plt.clf()
# plt.plot(trainCost, label = 'Training Cost')
# plt.plot(testCost, label = 'Testing Cost')
# plt.xlabel("Epoch")
# plt.ylabel("Cost")
# plt.legend()
# plt.show()
#
# pickle_file = 'FirstLayerWeights.pickle'
#
# try:
#    f = open(pickle_file, 'wb')
#    save = {
#       'second_dataset': secondDataset,
#       'W1': W1,
#       'b1_encode': b1_encode,
#     }
#    pickle.dump(save, f, protocol = 2)
#    f.close()
# except Exception as e:
#     print('Unable to save data to', pickle_file, ':', e)
#     raise

######################## start training the second one ########################################

pickle_file = 'FirstLayerWeights.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  secondDataset = save['second_dataset']
  W1Layer1 = save['W1']
  b1Layer1 = save['b1_encode']
  del save  # hint to help gc free up memory

thirdDataset, W1Layer2, b1Layer2, trainCost = trainSecondLayer(secondDataset, network_architecture, batch_size=10, training_epochs=20, learning_rate=5.,
                                n_samples=5000, noise=0 )
# plot training cost and testing cost
# print(len(trainCost))
# plt.clf()
# plt.plot(trainCost, label = 'Training Cost')
# plt.xlabel("Epoch")
# plt.ylabel("Cost")
# plt.legend()
# plt.show()
# pickle_file = 'SecondLayerWeights.pickle'
#
# try:
#    f = open(pickle_file, 'wb')
#    save = {
#        'third_dataset': thirdDataset,
#        'W1Layer1': W1Layer1,
#        'W1Layer2' : W1Layer2,
#        'b1Layer1' : b1Layer1,
#        'b1Layer2': b1Layer2,
#     }
#    pickle.dump(save, f, protocol = 2)
#    f.close()
# except Exception as e:
#     print('Unable to save data to', pickle_file, ':', e)
#     raise
#
#
# ################################# Start training the classifier #####################################
#
# pickle_file = 'SecondLayerWeights.pickle'
#
# with open(pickle_file, 'rb') as f:
#   save = pickle.load(f)
#   thirdDataset = save['third_dataset']
#   W1Layer1 = save['W1Layer1']
#   b1Layer1 = save['b1Layer1']
#   W1Layer2 = save['W1Layer2']
#   b1Layer2 = save['b1Layer2']
#   del save  # hint to help gc free up memory


trainClassifer(thirdDataset, batch_size=100, training_epochs=10, learning_rate=0.001, decay_rate=0.99,
                                 n_samples=5000, noise=0)

####################################### on testing set ##################################

b1 = np.tile(b1_encode,(n_samples,1))
    third_dataset = np.tanh(np.add(np.dot(second_dataset[0:n_samples], W1), b1))
# print(len(trainCost))
# print(len(testCost))
# plt.clf()
# plt.plot(trainCost, label = 'Training Cost')
# plt.plot(testCost, label = 'Testing Cost')
# plt.xlabel("Epoch")
# plt.ylabel("Cost")
# plt.legend()
# plt.show()