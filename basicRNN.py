import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tqdm import tqdm
from word2vec2 import *

wordvec_dim	= 128
num_features = 20
sentence_length = 300
output_dim = 4
learning_rate = 1.0
vocabulary_size = 50000
batch_size = 30
validation_size = 1000

training_steps=5000
display_step=100
validation_step=100

posfileList = [i for i in range(12500)]
negfileList = [i for i in range(12500)]

def zero_pad(lst, length):
	answer = ['UNK']*(length-len(lst))
	answer.extend(lst)
	return answer

def down_sample(lst, length):
	to_keep = random.sample([i for i in range(len(lst))],length)
	answer = []
	for i in range(len(lst)):
		if i in to_keep:
			answer.append(lst[i])
	return answer


def read_review(filename):
	global sentence_length
	"""Extract the text enclosed in a zip file as a list of words."""
	if not os.path.exists(filename):
		print("File doesnt exist")
		return None
	with open(filename, 'r', encoding="utf-8") as f:
		data = f.read().replace('.','').replace('<br /><br />',' ').replace('(','').replace(')','').lower().split()
	if len(data) > sentence_length:
		return down_sample(data,sentence_length)
	elif len(data) < sentence_length:
		return zero_pad(data, sentence_length)
	else:
		return data

def convert_words(words, dictionary):
	converted = []
	for word in words:
		converted.append(dictionary.get(word,0))
	return converted

def complete_path(path):
	for i in range(11):
		if os.path.exists(path+str(i)+'.txt'):
			return path+str(i)+'.txt', i

def random_path(data_location):
	global posfileList, negfileList
	if len(posfileList)==0 and len(negfileList)==0:
		posfileList = [i for i in range(12500)]
		negfileList = [i for i in range(12500)]
		typ = random.choice(['pos/','neg/'])
		print("Exhausted all reviews")
	elif len(posfileList)==0:
		typ = 'neg/'
	elif len(negfileList)==0:
		typ = 'pos/'
	else:
		typ = random.choice(['pos/','neg/'])
	
	if typ == 'pos/':
		j = random.choice(posfileList)
		posfileList.remove(j)
	if typ == 'neg/':
		j = random.choice(negfileList)
		negfileList.remove(j)

	path, rating = complete_path(data_location+typ+str(j)+'_')
	if rating < 3:
		rating2 = 1
	elif rating < 6:
		rating2 = 2
	elif rating < 9:
		rating2 = 3
	else: 
		rating2 = 4 
	return path, rating2

def generate_batch(batch_size, dictionary, data_location):
	batch, labels = [], []
	for i in tqdm(range(batch_size)):
		label = [0]*output_dim
		path, rating = random_path(data_location)
		data = convert_words(read_review(path),dictionary)
		batch.extend(data)
		label[rating-1] = 1
		labels.extend([label])
	return(np.array(batch).reshape(batch_size,sentence_length), np.array(labels))

if __name__ == '__main__':
	filename = maybe_download('text8.zip', 31344016)
	vocabulary = read_data(filename)
	print('Number of words: ', len(vocabulary))
	data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
	                                                            vocabulary_size)
	del vocabulary  # Hint to reduce memory.

	summary = tf.summary.FileWriter('./saved/')
	graph = tf.Graph()
	wordVec = np.loadtxt('./wordvec.csv', delimiter=',')
	embed_init = tf.constant_initializer(wordVec)
	print("Embedding matrix loaded.", wordVec.shape)

	with graph.as_default():
		X = tf.placeholder(tf.int32, [None, sentence_length], name = 'X')
		Y = tf.placeholder("float", [None, output_dim], name = 'Y')
		embedding = tf.get_variable('embedding', shape=[vocabulary_size, wordvec_dim], initializer = embed_init)
		embeddedInput = tf.nn.embedding_lookup(embedding, X)

		weights = {'out':tf.Variable(tf.random_normal([num_features, output_dim]))}
		biases = {'out': tf.Variable(tf.random_normal([output_dim]))}

		def RNN(x, weights, biases):

		    # Prepare data shape to match `rnn` function requirements
		    # Current data input shape: (batch_size, timesteps, n_input)
		    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

		    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
		    x = tf.unstack(x, sentence_length, 1)

		    # Define a lstm cell with tensorflow
		    lstm_cell = rnn.LSTMCell(num_features, forget_bias=1.0)
		    forgetful_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.8, input_keep_prob=0.8)
		    # Get lstm cell output
		    outputs, states = rnn.static_rnn(forgetful_cell, x, dtype=tf.float32)

		    # Linear activation, using rnn inner loop last output
		    return tf.matmul(outputs[-1], weights['out']) + biases['out']


		logits = RNN(embeddedInput, weights, biases)
		prediction = tf.nn.softmax(logits)
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	    logits = logits, labels = Y))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
		optimizer2 = tf.train.AdadeltaOptimizer(learning_rate = 1.0)
		train_op = optimizer2.minimize(loss_op)

		# Evaluate model (with test logits, for dropout to be disabled)
		correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# Initialize the variables (i.e. assign their default value)
		init = tf.global_variables_initializer()

		summary.add_graph(graph)

		with tf.Session() as sess:

		    # Run the initializer
		    sess.run(init)
		    with open('./logs/log2.txt', 'w', 1) as f:
			    for step in range(1, training_steps+1):
			        batch_x, batch_y = generate_batch(batch_size, dictionary, 'train/')
			        # Run optimization op (backprop)
			        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
			        if step % display_step == 0 or step == 1:
			            # Calculate batch loss and accuracy
			            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
			                                                                 Y: batch_y})
			            print("Step " + str(step) + ", Minibatch Loss= " + \
			                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
			                  "{:.3f}".format(acc))
			            f.write("{:.4f}".format(loss) + ", " + "{:.3f}".format(acc))
			        if step % validation_step == 0 or step == 1:
				        valid_x, valid_y = generate_batch(validation_size, dictionary, 'test/')
				        val_acc = sess.run(accuracy, feed_dict={X: valid_x, Y: valid_y})
				        print("Unseen accuracy for "+str(validation_size)+" reviews: "+"{:.3f}".format(val_acc*100)+"%")
				        f.write(", " + "{:.3f}\n".format(val_acc))
		    print("Optimization Finished!")

		saver = tf.train.Saver(weights)

