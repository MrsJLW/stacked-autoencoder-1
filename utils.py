import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def reformat(dataset, labels, num_labels, image_size):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]                               
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

#get notMNIST data
def getnotMNISTData(image_size, num_labels, pickle_file):
    
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_d = save['train_dataset']
        train_l = save['train_labels']
        reserve_dataset = save['test_dataset']
        reserve_labels = save['test_labels']
        del save  # hint to help gc free up memory

    train_dataset = train_d[:40000]
    train_labels = train_l[:40000]
    test_dataset = train_d[40000:45000]
    test_labels = train_l[40000:45000]
    valid_dataset = train_d[45000:50000]
    valid_labels = train_l[45000:50000]

    print ('Training set', train_dataset.shape, train_labels.shape)                     
    print ('Validation set', valid_dataset.shape, valid_labels.shape)                  
    print ('Test set', test_dataset.shape, test_labels.shape)                           

    
    train_dataset, train_labels = reformat(train_dataset, train_labels, num_labels, image_size)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, num_labels, image_size)
    test_dataset, test_labels = reformat(test_dataset, test_labels, num_labels, image_size)
    print ('Training set', train_dataset.shape, train_labels.shape, num_labels, image_size)
    print ('Validation set', valid_dataset.shape, valid_labels.shape)
    print ('Test set', test_dataset.shape, test_labels.shape)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar





####http://deeplearning.net/tutorial/code/utils.py
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

def printWeights(w, filename):
    im_dim = np.sqrt(w.shape[0])
    print('saving weights')
     # output hidden Layer
    image = Image.fromarray(tile_raster_images(X=w.T,
                            img_shape=(im_dim, im_dim), tile_shape=(10, 10),
                            tile_spacing=(1, 1)))
    image.save(filename)


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


# plot training cost and testing cost
def plotTraining(trainCost, testCost, filename):
    plt.clf()
    plt.plot(trainCost, label = 'Training Cost')
    plt.plot(testCost, label = 'Validation Cost')
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.savefig(filename)