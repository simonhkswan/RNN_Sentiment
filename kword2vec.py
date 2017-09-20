from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import tensorflow as tf
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

from keras.models import Sequential,Model
from keras.optimizers import RMSprop
from keras.layers import Embedding,LSTM,Dense,Lambda,merge,Input
from keras.callbacks import TensorBoard,ModelCheckpoint,Callback
from keras import backend as K

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

# Step 3: Function to generate a training batch for the skip-gram model.

filename = maybe_download('text8.zip', 31344016)
vocabulary = read_data(filename)
print('Number of words: ', len(vocabulary))

def generate_batches(data, size, contextWidth, negativeSize):
  cHalfWidth = int(contextWidth/2)
  words = []
  contexts = []
  negatives = []
  index = random.sample(range(cHalfWidth,len(data)-cHalfWidth),size)
  for z in tqdm(index):
      context = []
      for m in range(-cHalfWidth,cHalfWidth+1):
        if m == 0:
          words.append([data[z]])
        else: 
          context.append(data[z+m])
      contexts.append(context)
      negatives.append(random.sample(data,negativeSize))
  return([np.array(words),np.array(contexts),np.array(negatives)],[np.array([1]*size),np.array([[0]*negativeSize]*size)])

  # def generate_batches(data, size, contextWidth, negativeSize, repeats):
  # numword = repeats
  # cHalfWidth = int(contextWidth/2)
  # words = []
  # contexts = []
  # negatives = []
  # npdata = np.array(data)
  # for z in tqdm(range(size)):
  #   wordIndex = z%vocabulary_size
  #   (ii,) = np.where(npdata.astype(int) == wordIndex) 
  #   q = np.random.choice(ii,numword)
  #   for j in q:
  #     context = []
  #     for m in range(-cHalfWidth,cHalfWidth+1):
  #       if m == 0:
  #         words.append([data[j]])
  #       else: 
  #         context.append(data[j+m])
  #     contexts.append(context)
  #     negatives.append(random.sample(data,negativeSize))
  # return([np.array(words),np.array(contexts),np.array(negatives)],[np.array([1]*size*numword),np.array([[0]*negativeSize]*size*numword)])


vocabulary_size = 5000
data_index = 0
batch_size = 128
wordvec_dim = 32
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 3       # How many words to consider left and right.
num_skips = 4         # How many times to reuse an input to generate a label.
context_half = 3
context_size = context_half*2

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
neg_size = 5    # Number of negative examples to sample.


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,vocabulary_size)
del vocabulary  # Hint to reduce memory.

X,Y = generate_batches(data, 5000000, context_size, neg_size)
vX, vY = generate_batches(data, 5000, context_size, neg_size)

word = Input(shape=(1,))
context = Input(shape=(context_size,))
negSamples = Input(shape=(neg_size,))

word2vec = Embedding(input_dim=vocabulary_size,output_dim=wordvec_dim, embeddings_initializer='glorot_normal', name='word2vec')

vec_word = word2vec(word)
vec_context = word2vec(context)
vec_negSamples = word2vec(negSamples)
cbow = Lambda(lambda x: K.mean(x, axis=1))(vec_context)

word_context = merge([vec_word, cbow], mode='dot')
negative_context = merge([vec_negSamples, cbow], mode='dot', concat_axis=-1)

model = Model(input=[word,context,negSamples], output=[word_context,negative_context])
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
#model.summary()
tensorboard = TensorBoard(log_dir='./wordveclogs/8wordvec', 
  batch_size=500, histogram_freq=1, write_images=True, write_grads=False, write_graph=True, embeddings_freq=1)
model_checkpoint = ModelCheckpoint('./wordveclogs/8wordvec/model.hdf5')
model.fit(X,Y,epochs=50,batch_size=500,callbacks=[model_checkpoint,tensorboard], validation_data=(vX,vY))

