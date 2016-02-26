import pickle
import numpy as np

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
