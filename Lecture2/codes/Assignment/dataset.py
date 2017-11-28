import numpy as np
from scipy import ndimage
from os import listdir

def load_dataset():
      train_X = []
      train_Y = []
      data_files = ["train/"+f for f in listdir("train") ]

      data_files.sort()
      c=0;  n=0
      for f in data_files:
          if "non" in f:  
             train_Y.append(0)
             n+=1
          else: 
             train_Y.append(1)
             c+=1
      print ("# of train samples= ",len(data_files),"# of cats= ",c,", # of non-cats= ",n)

      for fname in data_files:
          image = np.array(ndimage.imread(fname, mode="RGB"))
          train_X.append(image)

      test_X = []
      test_Y = []
      test_files = ["test/"+f for f in listdir("test") ]
      c=0;  n=0
      for f in test_files:
          if "non-cat" in f:  
             test_Y.append(0)
             n+=1
          else: 
             test_Y.append(1)
             c+=1
      print ("# of test samples= ",len(test_files),"# of cats= ",c,", # of non-cats= ",n)

      for fname in test_files:
          image = np.array(ndimage.imread(fname, mode="RGB"))
          test_X.append(image)

      classes = ['non-cat', 'cat'] 
      return [np.array(train_X), np.array([train_Y]), np.array(test_X), np.array([test_Y]), np.array(classes)]
