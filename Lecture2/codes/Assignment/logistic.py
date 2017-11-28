import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from os import listdir
import dataset

# To load the dataset, write the function below
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = dataset.load_dataset()

# Example of a picture (make sure everything is working well)
index = 25
plt.imshow(train_set_x_orig[index])
plt.show()
import pdb; pdb.set_trace()
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[int(np.squeeze(train_set_y[:, index]))] +  "' picture.") 


