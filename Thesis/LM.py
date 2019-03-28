# coding: utf-8

from __future__ import print_function
from LM_HelpFunctions import *
from LM_PrintFunctions import *
import numpy as np
import tensorflow as tf
import collections
import time
import io

#np.set_printoptions(threshold=np.nan)



class LM:
    def __init__(self, training_file, n_hidden, embed_size, time_steps, batch_size, n_epochs,
                 dropout_rate = 0.5, learning_rate = 0.0001, all_words = True, min_nb_appearances = 5,
                 encoding = "utf8", unknown_string = "<UNK>", test_time_steps = 1, valid_file = None, voc_file = None,
                 xl_training = False, output_file = "output.txt"):
        self.training_file = training_file
        self.n_hidden = n_hidden
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.init_learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        if valid_file is None:
            valid_file = training_file
        self.valid_file = valid_file
        self.unknown_string = unknown_string
        self.encoding = encoding
        self.output_file = output_file

        print("Preparing the data..")
        print("Timer starts..")
        start1 = time.time()

        # Returns the sequence_range as well as a possibly updated time_steps
        def sequence_range(n_input, text_size, batch_size):
            seq_range = text_size // batch_size
            if seq_range - 1 < n_input:
                n_input = seq_range - 1
            return seq_range, n_input

        # Loading the data in an array list
        # Input: the name of the file
        # Output: a list of which each element represents a word of the text
        def read_data(file_name, encoding = "utf8"):
            with io.open(file_name, encoding=encoding) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            content = [word for i in range(len(content)) for word in content[i].split()]
            content = np.array(content)
            return content

        # training_data and test_data are a list of the text
        if xl_training is False:
            self.training_data = read_data(training_file, encoding=encoding)
        self.valid_data = read_data(valid_file, encoding=encoding)
        if voc_file is not None:
            voc_data = read_data(voc_file, encoding=encoding)
        else:
            voc_data = None

        print("Read the data: ", time.time() - start1)

        # Create a dictionary for the vocabulary, so that every word can be mapped to a real number.
        # We also create the reverse dictionary, so that a number can also be mapped to a word.
        # Input: the text as a list of words
        # Input: all_words: if false, only words with at last appearances appearances are considered known words
        # Input: appearances
        # Output: the dictionary and its reverse dictionary
        def build_data_set(train_words = None, all_words = True, appearances = 5,
                           unknown_string ="<unk>", voc_data = None):

            dictionary = dict()
            # vocabulary still have to be created
            if voc_data is None:
                print("Building data set..")
                print("Looking for the most common words..")
                count = collections.Counter(train_words).most_common()
                print("Found them!")
                appearances = 0 if all_words else appearances
                print("Minimal number of appearances: ", appearances)
                removed_words = 0
                print("Let's check which words have to be removed!")
                if appearances == 0:
                    for word, word_counts in count:
                        dictionary[word] = len(dictionary)
                else:
                    for word, word_counts in count:
                        if word_counts >= appearances:
                            dictionary[word] = len(dictionary)
                        else:
                            removed_words += 1
                print("Number of words removed: ", removed_words)
            # Forming the vocabulary is quite easy
            else:
                for i in range(len(voc_data)):
                    dictionary[voc_data[i]] = len(dictionary)

            print("Forming the dictionary was quite easy")
            # We add the unknown_string if it was not yet part of the dictionary
            if unknown_string not in dictionary.keys():
                dictionary[unknown_string] = len(dictionary)
            #reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
            return dictionary

        # Building the dictionary consisting of the id's
        if xl_training is False:
            self.dictionary = build_data_set(train_words=self.training_data, all_words=all_words, appearances=min_nb_appearances,
                             unknown_string=self.unknown_string, voc_data=voc_data)
        else:
            self.dictionary = build_data_set(all_words=all_words, appearances=min_nb_appearances,
                                             unknown_string=self.unknown_string, voc_data=voc_data)

        self.test_data = self.valid_data

        print("Built the data set: ", time.time() - start1)
        print("Setting up everything else..")

        # The length of the dictionary and the length of the text
        self.voc_size = len(self.dictionary)
        if xl_training is False:
            self.text_size = len(self.training_data)
        self.valid_text_size = len(self.valid_data)
        self.test_text_size = len(self.test_data)

        # sequence_range and updated time_steps
        if xl_training is False:
            self.seq_range, self.time_steps = sequence_range(time_steps, self.text_size, self.batch_size)
        else:
            self.time_steps = time_steps
        # initializing the time steps for the test phase
        self.test_time_steps = test_time_steps

        # tf graph input, i.e. the placeholders
        #  - x will be the input of the model. It is of size [batch_size, None] (since we don't know the time_steps for sure)
        #  - y will be the output of the model. It is of size [None, batch_size, voc_size].
        #  - seq_length is an array of [batch_size] indicating the time_steps for each batch element
        self.x = tf.placeholder(tf.int32, [self.batch_size, None])
        self.y = tf.placeholder(tf.int32, [None, self.batch_size, self.voc_size])
        # dropout should also be a placeholder, since we don't want any dropout when we are training.
        self.dropout = tf.placeholder(tf.float32)

        # We initialize the embedding matrix as a trainable variable, originally containing random values between -1 and 1
        # embed returns the columns of the input words x contains.
        # embed is of size [batch_size, time_steps, embed_size]
        self.embed_matrix = tf.Variable(tf.random_uniform((self.voc_size, self.embed_size), -1, 1))
        self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.x)

        # RNN output weights and biases: we initialize them with random values
        self.weights = {'out': tf.Variable(tf.random_normal([self.n_hidden, self.voc_size]))}
        self.biases = {'out': tf.Variable(tf.random_normal([self.voc_size]))}

        print("LM cell is ready: ", time.time() - start1)

    '''
    Prepares the RNN cell, states, outputs, prediction, accuracy and loss
    '''
    def __call__(self, training=True):

        print("Setting up __call__(self). Another timer started")
        start2 = time.time()

        # Training or no training:
        if training is True:
            time_steps = self.time_steps
        else:
            time_steps = self.test_time_steps

        # Returns a LSTM cell and its initial state
        def LSTM_cell(n_hidden, dropout_rate, batch_size):
            # We create a new basic 1-layer LSTM cell consisting of n_hidden nodes.
            # Note that the forget gate is automatically initialized to 1, which means we do not forget initially
            # We also add dropout at a dropout_rate to prevent the model from overfitting
            rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=dropout_rate)
            # We initialize the state:
            initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

            return rnn_cell, initial_state

        # Returns the outputs and states of the LSTM, as well as the output for each output word (pred)
        def RNN(embed, weights, biases, rnn_cell, init_states, time_steps):
            # Dynamic RNN runs the LSTM: we give embed as input
            outputs, states = tf.nn.dynamic_rnn(rnn_cell, embed, initial_state=init_states, dtype=tf.float32)
            # pred will contain the output for each input word of each batch
            pred = []
            for i in range(time_steps):
                try:
                    pred.append(tf.matmul(outputs[:, i], weights['out']) + biases['out'])
                except:
                    break

            return pred, outputs, states

        print("Finally setting up LSTM cell and RNN")
        self.cell, self.init_states = LSTM_cell(self.n_hidden, self.dropout, self.batch_size)
        self.pred, self.outputs, self.states = RNN(self.embed, self.weights, self.biases, self.cell, self.init_states,
                                                   time_steps)
        print("Did that! After: ", time.time() - start2)

        # Computing the cost using softmax with cross entropy and minimizing the cost using Adam
        #self.global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(self.init_learning_rate, global_step=self.global_step,
        #                                           decay_steps=self.seq_range, decay_rate=0.95, staircase=True)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.init_learning_rate).minimize(self.cost)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

        print("Now also set up the optimizers: ", time.time() - start2)

        # Returns the accuracy
        def get_accuracy(pred, y):
            accuracy = 0
            for i in range(len(pred)):
                correct_pred = (tf.equal(tf.argmax(pred[i], 1), tf.argmax(y[i], 1)))
                accuracy += tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            accuracy = accuracy / len(pred)
            return accuracy

        # Computing the accuracy
        self.accuracy = get_accuracy(self.pred, self.y)

        print("Accuracy computed: ", time.time() - start2)

        # Initializing the variables
        self.init = tf.global_variables_initializer()


    '''
    Function to validate the current trained model.
    It requires the following inputs:
        - sess: tf.Session()
    '''
    def valid(self, sess):



        stop, step = 0, 0
        loss_total = 0
        acc_total = 0
        offsets = []

        final_state = [self.init_states.c.eval(), self.init_states.h.eval()]

        while True:
            input_batch, output_batch, offsets, old_offsets, stop, _ \
                = next_batch(offsets, stop, self.time_steps, self.batch_size, self.valid_data,
                             self.voc_size, self.valid_text_size, self.dictionary, self.unknown_string)
            acc, loss, output, onehot_pred, final_state = sess.run([self.accuracy, self.cost, self.outputs, self.pred,
                                                                    self.states],
                    feed_dict={self.x: input_batch, self.y: output_batch, self.init_states: final_state,
                                                                                        self.dropout: 1.0})
            acc_total += acc
            loss_total += loss
            step += 1
            if stop == 1:
                break
        print_acc_loss_ppl(loss_total, acc_total, step)


    '''
    Function to train the RNN LM based on LSTM.
    It requires the following inputs: 
        - sess: tf.Session()
    Other input arguments are not compulsory. 
    If a saved model should be restored, restore_saved_model should be set to True. 
    In that case, saved_model_dir is compulsory.
    '''
    def train(self, sess, epoch_print_step = None, epoch_saver_step = 1,
              saver_dir ="lm_model_5", saved_model_dir = None, restore_saved_model = False):

        print("Started training..")
        print("Timer started..")
        start3 = time.time()

        print("Initializing self(training=True)")
        # Preparing the model
        self(training=True)

        print("self() set up after: ", time.time() - start3)

        print("Restoring a saved model if necessary..")
        # Restoring the saved model if necessary:
        if restore_saved_model is True and saved_model_dir is not None:
            print("Indeed restoring a model")
            loader = tf.train.Saver()
            loader.restore(sess, saved_model_dir)
        else:
            print("Did not need to restore a model")
            sess.run(self.init)

        print("Restored a model (or not) after: ", time.time() - start3)

        print("Preparing the training..")
        old_time = time.time()
        if epoch_print_step is None:
            epoch_print_step = 1
        print("Initializing the saver and already saving the model..")

        # Initializing a saver
        saver = tf.train.Saver()
        saver.save(sess, "steven/models/" + saver_dir + ".ckpt")
        print("Saved the model: ", time.time() - start3)
        print("Setting all parameters equal to 0 or initializing them with empty values..")
        iter_epoch, epoch, iter_total = 0, 0, 0
        # We define the offsets:
        offsets = []
        # Accuracy and loss per epoch
        acc_epoch, loss_epoch = 0, 0
        # Total accuracy and loss
        loss_total, acc_total = 0, 0
        # Initializing the final state
        final_state = [self.init_states.c.eval(), self.init_states.h.eval()]
        # step since last epoch update
        print("Starting the iterations..")
        iter_start = time.time()
        #iter_tot_time = time.time()
        # Iterating..
        train_time = time.time()
        while epoch < self.n_epochs:
            # Generating the input for x and y
            #print("Epoch = ", epoch)
            input_batch, output_batch, offsets, old_offsets, epoch, epoch_update \
                = next_batch(offsets, epoch, self.time_steps, self.batch_size, self.training_data,
                             self.voc_size, self.text_size, self.dictionary, self.unknown_string)
            #print("Finished next_batch after: ", time.time() - iter_start)
            # Running the model
            # If poor performance, add one_hot_pred and self.pred
            _, acc, loss, _, output, final_state = sess.run([self.optimizer, self.accuracy, self.cost,
                    self.pred, self.outputs, self.states], feed_dict={self.x: input_batch, self.y: output_batch,
                    self.init_states: final_state, self.dropout: self.dropout_rate})
            # Add the newly computed loss and accuracy to the loss and accuracy for the current epochs or total
            loss_epoch += loss
            acc_epoch += acc
            acc_total += acc
            loss_total += loss
            # Update iteration counters
            iter_epoch += 1
            iter_total += 1
            if iter_epoch % 50 == 0 and iter_epoch > 0:
                print("     Iteration: ", iter_epoch)
            # We print everything every epoch_print epochs:
            if epoch_update is True and epoch != 0 and epoch % epoch_print_step == 0:
                acc_epoch, loss_epoch = print_output_RNN_epoch(iter_epoch, loss_epoch, acc_epoch, epoch)
                iter_epoch = 0
            if epoch_update is True and epoch != 0 and epoch % epoch_saver_step == 0:
                saver.save(sess, "steven/models/" + saver_dir + ".ckpt")
            if epoch_update is True and epoch % 4 == 0:
                self.valid(sess)
                #print("Validated the model: ", time.time() - iter_start)
            #print("Total time: ", time.time() - iter_tot_time)
        print_final_stats(loss_total, acc_total, iter_total)
        print_time(time.time() - old_time)


    '''
    Function to train the RNN LM based on LSTM in case the training set is too large to read at once.
    It requires the following inputs: 
        - sess: tf.Session()
    Other input arguments are not compulsory. 
    If a saved model should be restored, restore_saved_model should be set to True. 
    In that case, saved_model_dir is compulsory.
    
     
    '''
    def train_xl(self, sess, saver_dir ="lm_model_5", saved_model_dir = None, restore_saved_model = False,
                 start = 1):

        # Loading the data in an array list
        # Input: the name of the file
        # Output: a list of which each element represents a word of the text
        def read_data(file_name):
            with io.open(file_name, encoding=self.encoding) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            content = [word for i in range(len(content)) for word in content[i].split()]
            content = np.array(content)
            return content

        # Help function to make it easier to write to the file
        def write(string):
            write_to_file(string, self.output_file)

        print("Started training..")
        print("Timer started..")
        start3 = time.time()

        print("Initializing self(training=True)")
        # Preparing the model
        self(training=True)

        print("self() set up after: ", time.time() - start3)

        print("Restoring a saved model if necessary..")
        # Restoring the saved model if necessary:
        if restore_saved_model is True and saved_model_dir is not None:
            print("Indeed restoring a model")
            loader = tf.train.Saver()
            loader.restore(sess, saved_model_dir)
        else:
            print("Did not need to restore a model")
            sess.run(self.init)

        print("Restored a model (or not) after: ", time.time() - start3)

        print("Preparing the training..")
        old_time = time.time()
        print("Initializing the saver and already saving the model..")

        # Initializing a saver
        saver = tf.train.Saver()
        saver.save(sess, "steven/models/" + saver_dir + ".ckpt")
        print("Saved the model: ", time.time() - start3)
        print("Setting all parameters equal to 0 or initializing them with empty values..")
        iter_epoch, epoch, iter_total = 0, 0, 0
        # Accuracy and loss per epoch
        acc_epoch, loss_epoch = 0, 0
        # Total accuracy and loss
        loss_total, acc_total = 0, 0
        # Initializing the final state
        final_state = [self.init_states.c.eval(), self.init_states.h.eval()]
        # step since last epoch update
        print("Starting the iterations..")
        write("Starting the iterations..")
        #iter_tot_time = time.time()
        # Iterating..
        while epoch < self.n_epochs:
            print("Epoch: ", epoch)
            write("Epoch: " + str(epoch))
            for i in range(start, 100):
                if i >= 10:
                    training_file = "news.en-000" + str(i) + "-of-00100"
                else:
                    training_file = "news.en-0000" + str(i) + "-of-00100"
                print("Reading: ", training_file)
                write("Reading: " + training_file)
                training_data = read_data("steven/data/" + training_file)
                text_size = len(training_data)
                stop = 0
                iter_epoch_file = 0
                # We define the offsets:
                offsets = []
                write("Training this file..")
                while True:
                    # Generating the input for x and y
                    #print("Epoch = ", epoch)
                    input_batch, output_batch, offsets, old_offsets, stop, epoch_update \
                        = next_batch(offsets, stop, self.time_steps, self.batch_size, training_data,
                                     self.voc_size, text_size, self.dictionary, self.unknown_string)
                    #print("Finished next_batch after: ", time.time() - iter_start)
                    # Running the model
                    # If poor performance, add one_hot_pred and self.pred
                    _, acc, loss, _, output, final_state = sess.run([self.optimizer, self.accuracy, self.cost,
                            self.pred, self.outputs, self.states], feed_dict={self.x: input_batch, self.y: output_batch,
                            self.init_states: final_state, self.dropout: self.dropout_rate})
                    # Add the newly computed loss and accuracy to the loss and accuracy for the current epochs or total
                    loss_epoch += loss
                    acc_epoch += acc
                    acc_total += acc
                    loss_total += loss
                    # Update iteration counters
                    iter_epoch_file += 1
                    iter_epoch += 1
                    iter_total += 1
                    if iter_epoch_file % 50 == 0 and iter_epoch_file > 0:
                        print("     Iteration: ", iter_epoch_file)
                    if stop == 1:
                        break
                saver.save(sess, "steven/models/" + saver_dir + ".ckpt")
                write("Finished and saved at epoch: " + str(epoch) + ", file: " + training_file)
                print_output_RNN_epoch(iter_epoch, loss_epoch, acc_epoch, epoch, file=self.output_file)
            epoch += 1
            start = 1
            acc_epoch, loss_epoch = print_output_RNN_epoch(iter_epoch, loss_epoch, acc_epoch, epoch, file=self.output_file)
            iter_epoch = 0
            saver.save(sess, "steven/models/" + saver_dir + ".ckpt")
            self.valid(sess)
            #print("Validated the model: ", time.time() - iter_start)
            #print("Total time: ", time.time() - iter_tot_time)
        print_final_stats(loss_total, acc_total, iter_total)
        print_time(time.time() - old_time)



    '''
    Function to test the RNN LM based on LSTM.
    It requires the following inputs:
        - sess: tf.Session()
    Other input arguments are not compulsory.
    '''
    def test(self, sess, restore_dir ="lm_model.ckpt", states_to_file = False, local_time_steps = 1):

        print("Until here everything is fine..")

        print("voc size: ", self.voc_size)
        print("text size: ", self.test_text_size)

        print("Here I am, but what should I do now?")

        print("Initialize self(training=False), i.e. LSTM cell")
        self(training=False)
        print("Prepare the saver")
        saver = tf.train.Saver()
        print("Initialize the parameters with zero or empty values")
        offsets = []
        stop, step = 0, 0
        loss_total = 0
        acc_total = 0
        states_h_dict = {}
        states_c_dict = {}
        word_count = {}
        word_count_embed = {}
        embed_dict = {}

        print("Restore the model..")
        saver.restore(sess, "steven/models/" + restore_dir)
        print("Model restored!")
        final_state = [self.init_states.c.eval(), self.init_states.h.eval()]
        print("Start the loop..")
        while True:
            #print("Running next_batch..")
            input_batch, output_batch, offsets, old_offsets, stop, _ \
                = next_batch(offsets, stop, local_time_steps, self.batch_size, self.test_data,
                             self.voc_size, self.test_text_size, self.dictionary, self.unknown_string)
            #print("Next batch returned!")
            #print("Running sess.run()..")
            acc, loss, output, _, final_state, embedding = sess.run([self.accuracy, self.cost, self.outputs,
                                                                               self.pred, self.states, self.embed],
                            feed_dict={self.x: input_batch, self.y: output_batch, self.init_states: final_state,
                                                                    self.dropout: 1.0})
            #print("sess.run() finished!")


            #print("Writing the states to a file..")
            # Write the states to a file:
            if states_to_file:
                states_c_dict, states_h_dict, word_count = update_states(states_c_dict, states_h_dict, word_count,
                                                                final_state, self.batch_size, self.test_data, offsets)
                embed_dict, word_count_embed = update_embbeding(embed_dict, word_count_embed, embedding,
                                                                self.batch_size, self.test_data, offsets)
            #print("Writing the states to file finished!")

            #if states_to_file:
            #    write_states_to_file(states_filename, final_state, self.batch_size, self.test_data, offsets)

            #print_predicted_words(batch_size=self.batch_size, training_data=self.test_data, old_offsets=old_offsets,
            #                      n_input=local_time_steps, reverse_dictionary=self.reverse_dictionary,
            #                      onehot_pred=onehot_pred)

            #print("Calculating the accuracy and loss..")
            acc_total += acc
            loss_total += loss
            step += 1
            if step % 50 == 0 and step > 0:
                print("     Iteration finished: ", step)
            if stop == 1:
                break

        print("Testing finished!")
        print_acc_loss_ppl(loss_total, acc_total, step)
        #if states_to_file:
        #    ask_questions("questions.txt", states_c_dict, states_h_dict)


