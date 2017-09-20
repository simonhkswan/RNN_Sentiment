import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Embedding,LSTM,Dense,Conv1D,MaxPooling1D
from keras.callbacks import TensorBoard,ModelCheckpoint,Callback
from basicRNN import *
import json

#embedWeights = np.loadtxt('./wordvec.csv', delimiter=',')
#print(embedWeights.shape)

rnn = Sequential()
rnn.add(Embedding(5000,32,input_length=300, trainable=True, name='word2vec'))
#rnn.add(Conv1D(200,2,padding='same'))
#rnn.add(MaxPooling1D(4))
rnn.add(LSTM(100,return_sequences=False,dropout=0.5))
rnn.add(Dense(4,activation='softmax'))

print('Compiling Model...',end='')
rmsp = RMSprop(lr=0.01)
rnn.compile(optimizer=rmsp,loss='categorical_crossentropy',metrics=['accuracy'])
rnn.load_weights('./imdbwvlogs/1imdbwv/model.h5', by_name=True)
rnn.summary()
print('Done')

#filename = maybe_download('text8.zip', 31344016)
#vocabulary = read_data(filename)
#print('Number of words: ',len(vocabulary))
#data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,5000)
#del vocabulary  # Hint to reduce memory.
with open('embed_dict.txt','r') as f:
	dictionary = json.load(f)

print('Generating training data:')
batch_x, batch_y = generate_batch(20000, dictionary, 'train/')
print('Generating validation data:')
validationData = generate_batch(5000, dictionary, 'test/')

tensorboard = TensorBoard(log_dir='./logs/1Blstm', 
	batch_size=1000, histogram_freq=1, write_images=True, write_grads=False, write_graph=True, embeddings_freq=2)
model_checkpoint = ModelCheckpoint('./logs/1Blstm/model.h5')

rnn.fit(batch_x, batch_y, epochs=30, batch_size=1000, validation_data=validationData, callbacks=[tensorboard,model_checkpoint], initial_epoch=0)



