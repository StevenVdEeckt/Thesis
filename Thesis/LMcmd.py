#! /usr/bin/env python
# encoding: utf-8


from LM import LM
import tensorflow as tf
import sys
import datetime
import time

'''
Reads the command line and receives its arguments from there:

Options
    -train <file>
        Training file as a text file.
    -valid <file>
        File to validate the model, default is set to the training file
    -epochs <int> 
        Number of epochs, default is 40
    -time <int>
        Number of time steps, default is 10
    -batch <int> 
        Number of words per batch, default is 10
    -hidden <int> 
        Number of hidden neurons, default is 512
    -embedding <int> 
        Dimension of the input embedding, default is 100
    -all_words <int>
        1 if all words must be learned, else 0 (default is 1)
    -appearances <int>
        Minimal number of appearances per word if all_words was 0, default is 3
    -epoch_saver_step <int>
        Indicating how often the model is saved in a saver, default is 5
    -saver_dir <String>
        File to save the model, default is 'lm_model'
    -restore_saved_model <int>
        0 if no previously trained model should be reused, else 1 (default is 0)
    -saved_model_dir <file>
        File of the previously trained model

'''

def print_help():

    print()
    print("Options:")
    print("     -train <file>")
    print("         Training file as a text file.")
    print("     -valid <file>")
    print("         File to validate the model, default is set to the training file")
    print("     -voc <file>")
    print("         File containing the words that should be our vocabulary")
    print("     -epochs <int>")
    print("         Number of epochs, default is 40")
    print("     -time <int>")
    print("         Number of time steps, default is 10")
    print("     -batch <int>")
    print("         Number of words per batch, default is 10")
    print("     -hidden <int>")
    print("         Number of hidden neurons, default is 512")
    print("     -embedding <int>")
    print("         Dimension of the input embedding, default is 100")
    print("     -all_words <int>")
    print("         1 if all words must be learned, else 0 (default is 1)")
    print("     -appearances <int>")
    print("         Minimal number of appearances per word if all_words was 0, default is 3")
    print("     -epoch_saver_step <int>")
    print("         Indicating how often the model is saved in a saver, default is 5")
    print("     -saver_dir <String>")
    print("         File to save the model, default is 'lm_model'")
    print("     -restore_saved_model <int>")
    print("         0 if no previously trained model should be reused, else 1 (default is 0)")
    print("     -saved_model_dir <file>")
    print("         File of the previously trained model")
    print()
    print("NOTE: order of the commands must be respected!")




def read_cmd(args):

    training_file = None
    valid_file = None
    epochs = 40
    time_steps = 10
    batch_size = 10
    n_hidden = 512
    embed_size = 100
    all_words = True
    appearances = 5
    epoch_saver_step = 1
    saver_dir = "lm_model_" + str(round(time.time() % 500))
    restore_saved_model = False
    saved_model_dir = None
    voc_file = None

    # At position 0, we have LMcmd.py, so we start from 1
    iter = 1
    while iter < len(args):
        if args[iter] == "-train":
            training_file = args[iter+1]
            valid_file = args[iter+1]
        elif args[iter] == "-valid":
            valid_file = args[iter+1]
        elif args[iter] == "-voc":
            voc_file = args[iter+1]
        elif args[iter] == "-epochs":
            epochs = int(args[iter + 1])
        elif args[iter] == "-time":
            time_steps = int(args[iter+1])
        elif args[iter] == "-batch":
            batch_size = int(args[iter+1])
        elif args[iter] == "-hidden":
            n_hidden = int(args[iter+1])
        elif args[iter] == "-embedding":
            embed_size = int(args[iter+1])
        elif args[iter] == "-all_words":
            if int(args[iter+1]) == 0:
                all_words = False
            elif int(args[iter+1]) == 1:
                all_words = True
        elif args[iter] == "-appearances":
            appearances = int(args[iter+1])
        elif args[iter] == "-epoch_saver_step":
            epoch_saver_step = int(args[iter+1])
        elif args[iter] == "-saver_dir":
            saver_dir = args[iter+1]
        elif args[iter] == "-restore_saved_model":
            if int(args[iter+1]) == 0:
                restore_saved_model = False
            elif int(args[iter+1]) == 1:
                restore_saved_model = True
        elif args[iter] == "-saved_model_dir":
            saved_model_dir = args[iter+1]

        iter += 2

    # Write batch_size, n_hidden, embed_size, all_words, appearances to file
    with open("lm_cmd.txt", "w") as file:
        pass
        file.write("BATCH_SIZE = " + str(batch_size) + "\n")
        file.write("N_HIDDEN = " + str(n_hidden) + "\n")
        file.write("EMBED_SIZE = " + str(embed_size) + "\n")
        file.write("ALL_WORDS = " + str(all_words) + "\n")
        file.write("APPEARANCES = " + str(appearances) + "\n")
        file.write("TRAIN_FILE = " + str(training_file) + "\n")
        file.write("VOC_FILE = " + str(voc_file) + "\n")
        file.write("TIME_STEPS = " + str(time_steps) + "\n")

    print("batch_size: ", batch_size)
    print("time_steps: ", time_steps)
    print("embed_size: ", embed_size)
    print("n_hidden: ", n_hidden)
    print("saver_dir: ", saver_dir)
    return [training_file, epochs, time_steps, batch_size, n_hidden, embed_size, all_words, appearances,
            epoch_saver_step, saver_dir, restore_saved_model, saved_model_dir, valid_file, voc_file]


print("Program started at: ", datetime.datetime.now())
start1 = time.time()

if sys.argv[1] == "help" or sys.argv[1] == "HELP" or sys.argv[1] == "-help":
    print_help()

else:
    print("Reading the command line..")
    input = read_cmd(sys.argv)
    print("Read the command line!")
    print("Ready to set up LM cell after: ", time.time() - start1)

    print("Setting up the LM..")
    LSTM_cell = LM(training_file=input[0], n_epochs=input[1], time_steps=input[2], batch_size=input[3],
                   n_hidden=input[4], embed_size=input[5], all_words=input[6], min_nb_appearances=input[7],
                   valid_file=input[12], voc_file=input[13])

    print("LM cell set up after: ", time.time() - start1)

    print("Did that!")

    print("Voc size: ", LSTM_cell.voc_size)
    print("Text size: ", LSTM_cell.text_size)

    print("Starting the session..")
    with tf.Session() as sess:
        print("Starting training the model after: ", time.time() - start1)
        LSTM_cell.train(sess, epoch_saver_step=input[8], saver_dir=input[9],
                        saved_model_dir=input[11], restore_saved_model=input[10])



