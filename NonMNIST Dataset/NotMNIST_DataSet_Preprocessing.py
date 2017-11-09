#LOAD NON-MNIST DATASET

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans	
from time import time		
from sklearn import svm
		
 url = 'http://commondatastorage.googleapis.com/books1000/'
 last_percent_reported = None
 
 def download_progress_hook(count, blockSize, totalSize):
   """A hook to report the progress of a download. This is mostly intended for users with
   slow internet connections. Reports every 1% change in download progress.
   """
   global last_percent_reported
   percent = int(count * blockSize * 100 / totalSize)
 
   if last_percent_reported != percent:
     if percent % 5 == 0:
       sys.stdout.write("%s%%" % percent)
       sys.stdout.flush()
     else:
       sys.stdout.write(".")
       sys.stdout.flush()
 
     last_percent_reported = percent
 
 def maybe_download(filename, expected_bytes, force=False):
   """Download a file if not present, and make sure it's the right size."""
   if force or not os.path.exists(filename):
     print('Attempting to download:', filename) 
     filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
     print('\nDownload Complete!')
   statinfo = os.stat(filename)
   if statinfo.st_size == expected_bytes:
     print('Found and verified', filename)
   else:
     raise Exception(
       'Failed to verify ' + filename + '. Can you get to it with a browser?')
   return filename
 
 train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
 test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

''' 
Found and verified notMNIST_large.tar.gz
Found and verified notMNIST_small.tar.gz
'''

 
 def maybe_extract(filename, force=False):
   root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
   if os.path.isdir(root) and not force:
     # You may override by setting force=True.
     print('%s already present - Skipping extraction of %s.' % (root, filename))
   else:
     print('Extracting data for %s. This may take a while. Please wait.' % root)
     tar = tarfile.open(filename)
     sys.stdout.flush()
     tar.extractall()
     tar.close()
   data_folders = [
     os.path.join(root, d) for d in sorted(os.listdir(root))
     if os.path.isdir(os.path.join(root, d))]
   if len(data_folders) != num_classes:
     raise Exception(
       'Expected %d folders, one per class. Found %d instead.' % (
         num_classes, len(data_folders)))
   print(data_folders)
   return data_folders
 
 train_folders = maybe_extract(train_filename)
 test_folders = maybe_extract(test_filename)
 
notMNIST_large already present - Skipping extraction of notMNIST_large.tar.gz.
['notMNIST_large\\A', 'notMNIST_large\\B', 'notMNIST_large\\C', 'notMNIST_large\\D', 'notMNIST_large\\E', 'notMNIST_large\\F', 'notMNIST_large\\G', 'notMNIST_large\\H', 'notMNIST_large\\I', 'notMNIST_large\\J']
notMNIST_small already present - Skipping extraction of notMNIST_small.tar.gz.
['notMNIST_small\\A', 'notMNIST_small\\B', 'notMNIST_small\\C', 'notMNIST_small\\D', 'notMNIST_small\\E', 'notMNIST_small\\F', 'notMNIST_small\\G', 'notMNIST_small\\H', 'notMNIST_small\\I', 'notMNIST_small\\J']
'''
In [3]: Image(filename="notMNIST_small/A/Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png")
Out[3]: 
'''
￼
 

fn = os.listdir("notMNIST_small/A/")

for file in fn[:5]:
     path = 'notMNIST_small/A/' + file
     display(Image(path))
     

￼

￼

￼

￼

￼

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
 
 def load_letter(folder, min_num_images):
   """Load the data for a single letter label."""
   image_files = os.listdir(folder)
   dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                          dtype=np.float32)
   print(folder)
   num_images = 0
   for image in image_files:
     image_file = os.path.join(folder, image)
     try:
       image_data = (ndimage.imread(image_file).astype(float) - 
                     pixel_depth / 2) / pixel_depth
       if image_data.shape != (image_size, image_size):
         raise Exception('Unexpected image shape: %s' % str(image_data.shape))
       dataset[num_images, :, :] = image_data
       num_images = num_images + 1
     except IOError as e:
       print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
 
   dataset = dataset[0:num_images, :, :]
   if num_images < min_num_images:
     raise Exception('Many fewer images than expected: %d < %d' %
                     (num_images, min_num_images))
 
   print('Full dataset tensor:', dataset.shape)
   print('Mean:', np.mean(dataset))
   print('Standard deviation:', np.std(dataset))
   return dataset
 
 def maybe_pickle(data_folders, min_num_images_per_class, force=False):
   dataset_names = []
   for folder in data_folders:
     set_filename = folder + '.pickle'
     dataset_names.append(set_filename)
     if os.path.exists(set_filename) and not force:
       # You may override by setting force=True.
       print('%s already present - Skipping pickling.' % set_filename)
     else:
       print('Pickling %s.' % set_filename)
       dataset = load_letter(folder, min_num_images_per_class)
       try:
         with open(set_filename, 'wb') as f:
           pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
       except Exception as e:
         print('Unable to save data to', set_filename, ':', e)
 
   return dataset_names
 
 train_datasets = maybe_pickle(train_folders, 45000)
 test_datasets = maybe_pickle(test_folders, 1800)
 # With would automatically close the file after the nested block of code
 with open(pickle_file, 'rb') as f:
 
     # unpickle
     letter_set = pickle.load(f)  
 
     # pick a random image index
     sample_idx = np.random.randint(len(letter_set))
 
     # extract a 2D slice
     sample_image = letter_set[sample_idx, :, :]  
     plt.figure()
 
     # display it
     plt.imshow(sample_image)
     
def randomize(dataset, labels):
   permutation = np.random.permutation(labels.shape[0])
   shuffled_dataset = dataset[permutation,:,:]
   shuffled_labels = labels[permutation]
   return shuffled_dataset, shuffled_labels
 train_dataset, train_labels = randomize(train_dataset, train_labels)
 test_dataset, test_labels = randomize(test_dataset, test_labels)
 valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
 

def make_arrays(nb_rows, img_size):
    if nb_rows:
      dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
      labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
      dataset, labels = None, None
    return dataset, labels
  
  def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
  
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):       
      try:
        with open(pickle_file, 'rb') as f:
          letter_set = pickle.load(f)
          # let's shuffle the letters to have random validation and training set
          np.random.shuffle(letter_set)
          if valid_dataset is not None:
            valid_letter = letter_set[:vsize_per_class, :, :]
            valid_dataset[start_v:end_v, :, :] = valid_letter
            valid_labels[start_v:end_v] = label
            start_v += vsize_per_class
            end_v += vsize_per_class
  
          train_letter = letter_set[vsize_per_class:end_l, :, :]
          train_dataset[start_t:end_t, :, :] = train_letter
          train_labels[start_t:end_t] = label
          start_t += tsize_per_class
          end_t += tsize_per_class
      except Exception as e:
        print('Unable to process data from', pickle_file, ':', e)
        raise
  
    return valid_dataset, valid_labels, train_dataset, train_labels
  
  
  train_size = 5000
  valid_size = 100
  test_size = 100
  
  valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
  _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
  
  print('Training:', train_dataset.shape, train_labels.shape)
  print('Validation:', valid_dataset.shape, valid_labels.shape)
  print('Testing:', test_dataset.shape, test_labels.shape)
'''  
Training: (50000, 28, 28) (50000,)
Validation: (5000, 28, 28) (5000,)
Testing: (5000, 28, 28) (5000,)
'''

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


pickle_file = os.path.join(".", 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
  
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


'''
In [11]: test_dataset.shape
Out[11]: (5000, 28, 28)
'''
samples, width, height = train_dataset.shape
  X_train = np.reshape(train_dataset,(samples,width*height))
  y_train = train_labels
  
  # Prepare testing data
  samples, width, height = test_dataset.shape
  X_test = np.reshape(test_dataset,(samples,width*height))
  y_test = test_labels
  

  from sklearn.linear_model import LogisticRegression
  
  # Instantiate
  lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000, n_jobs=-1)
  
  # Fit
  lg.fit(X_train, y_train)
  
  # Predict
  y_pred = lg.predict(X_test)
  
  # Score
  from sklearn import metrics
  metrics.accuracy_score(y_test, y_pred)

'''  
[Parallel(n_jobs=-1)]: Done   1 out of   1 | elapsed: 10.5min finished
Out[13]: 0.90010000000000001
'''
notMNISTPickleDS= pickle.load( open( "notMNIST.pickle", "rb" ) )

from sklearn.cluster import KMeans

train_dataset = notMNISTPickleDS['train_dataset']
train_labels = notMNISTPickleDS['train_labels']
valid_dataset = notMNISTPickleDS['valid_dataset']
valid_labels  = notMNISTPickleDS['valid_labels']
test_dataset = notMNISTPickleDS['test_dataset']
test_labels = notMNISTPickleDS['test_labels']

# Prepare training data
samples, width, height = train_dataset.shape
X_train = np.reshape(train_dataset,(samples,width*height))
y_train = train_labels

# Prepare testing data
samples, width, height = test_dataset.shape
X_test = np.reshape(test_dataset,(samples,width*height))
y_test = test_labels

labels = y_train
data = X_train

#ScatterPlot after KMeans
'''
kmeans = KMeans(n_clusters=10, random_state=0).fit(X_train)

from sklearn.decomposition import PCA
from time import time
n_digits = 10
sample_size = 1000
labels = y_train
data = X_train
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.completeness_score(labels, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)


 reduced_data = PCA(n_components=n_digits).fit_transform(data)
kmeans = KMeans(n_clusters=n_digits, n_init=1)
kmeans.fit(reduced_data)

randomplotData = reduced_data[np.random.choice(reduced_data.shape[0], 7000, replace=False), :]

h = .02 
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

lt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(randomplotData[:, 0], randomplotData[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

'''
