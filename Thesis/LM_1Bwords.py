#! /usr/bin/env python
# encoding: utf-8


from LM import LM
import tensorflow as tf
import sys
import datetime
import time
from LM_HelpFunctions import write_to_file

'''
Reads the command line and receives its arguments from there:

Unlike in LMcmd.py, the training set, valid set as well as the vocabulary are already determined.

Options
    -epochs <int> 
        Number of epochs, default is 10
    -time <int>
        Number of time steps, default is 50
    -batch <int> 
        Number of words per batch, default is 40
    -hidden <int> 
        Number of hidden neurons, default is 512
    -embedding <int> 
        Dimension of the input embedding, default is 100
    -saver_dir <String>
        File to save the model, default is 'lm_model'
    -restore <file>
        File of the previously trained model
    -voc <file>
        File of the vocabulary - default is steven/voc100.txt
    -start <int>
        Start the first epoch by reading this file - not starting from 1, default is 1

'''

def print_help():

    print()
    print("Options:")
    print("     -epochs <int>")
    print("         Number of epochs, default is 10")
    print("     -time <int>")
    print("         Number of time steps, default is 50")
    print("     -batch <int>")
    print("         Number of words per batch, default is 40")
    print("     -hidden <int>")
    print("         Number of hidden neurons, default is 512")
    print("     -embedding <int>")
    print("         Dimension of the input embedding, default is 100")
    print("     -saver_dir <String>")
    print("         File to save the model, default is 'lm_model'")
    print("     -restore <file>")
    print("         File of the previously trained model")
    print("     -voc <file>")
    print("         File of the vocabulary - default is steven/voc100.txt")
    print("     -start <int>")
    print("         Start the first epoch by reading this file - not starting from 1, default is 1")
    print()
    print("NOTE: order of the commands must be respected!")




def read_cmd(args):

    training_file = ""
    valid_file = "1B_valid_100k.txt"
    epochs = 10
    time_steps = 50
    batch_size = 40
    n_hidden = 512
    embed_size = 100
    epoch_saver_step = 1
    saver_dir = "lm_model_" + str(round(time.time() % 500))
    restore_saved_model = False
    saved_model_dir = None
    voc_file = "steven/voc100.txt"
    start = 1

    # At position 0, we have LMcmd.py, so we start from 1
    iter = 1
    while iter < len(args):
        if args[iter] == "-epochs":
            epochs = int(args[iter + 1])
        elif args[iter] == "-time":
            time_steps = int(args[iter+1])
        elif args[iter] == "-batch":
            batch_size = int(args[iter+1])
        elif args[iter] == "-hidden":
            n_hidden = int(args[iter+1])
        elif args[iter] == "-embedding":
            embed_size = int(args[iter+1])
        elif args[iter] == "-epoch_saver_step":
            epoch_saver_step = int(args[iter+1])
        elif args[iter] == "-saver_dir":
            saver_dir = args[iter+1]
        elif args[iter] == "-restore":
            saved_model_dir = args[iter+1]
        elif args[iter] == "-voc":
            voc_file = args[iter+1]
        elif args[iter] == "-start":
            start = int(args[iter+1])
        iter += 2

    if saved_model_dir is not None:
        restore_saved_model = True


    # Write batch_size, n_hidden, embed_size, all_words, appearances to file
    with open("lm_cmd.txt", "w") as file:
        pass
        file.write("BATCH_SIZE = " + str(batch_size) + "\n")
        file.write("N_HIDDEN = " + str(n_hidden) + "\n")
        file.write("EMBED_SIZE = " + str(embed_size) + "\n")
        file.write("ALL_WORDS = " + "True" + "\n")
        file.write("APPEARANCES = " + "5" + "\n")
        file.write("TRAIN_FILE = " + str(training_file) + "\n")
        file.write("VOC_FILE = " + str(voc_file) + "\n")
        file.write("TIME_STEPS = " + str(time_steps) + "\n")

    print("batch_size: ", batch_size)
    print("time_steps: ", time_steps)
    print("embed_size: ", embed_size)
    print("n_hidden: ", n_hidden)
    print("saver_dir: ", saver_dir)
    all_words = False
    appearances = 5
    return [training_file, epochs, time_steps, batch_size, n_hidden, embed_size, all_words, appearances,
            epoch_saver_step, saver_dir, restore_saved_model, saved_model_dir, valid_file, voc_file, start]

file = "steven/LM_1Bwords" + str(round(time.time() % 500)) + ".out"
print("Program started at: ", datetime.datetime.now())
write_to_file("Program started at: " + str(datetime.datetime.now()), file)
start1 = time.time()

if len(sys.argv) > 1 and (sys.argv[1] == "help" or sys.argv[1] == "HELP" or sys.argv[1] == "-help"):
    print_help()

else:
    print("Reading the command line..")
    input = read_cmd(sys.argv)
    print("Read the command line!")
    print("Ready to set up LM cell after: ", time.time() - start1)

    print("Setting up the LM..")
    LSTM_cell = LM(training_file=input[0], n_epochs=input[1], time_steps=input[2], batch_size=input[3],
                   n_hidden=input[4], embed_size=input[5], all_words=input[6], min_nb_appearances=input[7],
                   valid_file=input[12], voc_file=input[13], output_file=file, xl_training=True)

    print("LM cell set up after: ", time.time() - start1)

    print("Did that!")

    print("Voc size: ", LSTM_cell.voc_size)

    print("Starting the session..")
    with tf.Session() as sess:
        print("Starting training the model after: ", time.time() - start1)
        write_to_file("Starting training the model after: " + str(time.time() - start1), file)
        LSTM_cell.train_xl(sess, saver_dir=input[9],
                        saved_model_dir=input[11], restore_saved_model=input[10], start=input[14])



