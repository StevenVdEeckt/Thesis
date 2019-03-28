#! /usr/bin/env python
# encoding: utf-8


from LM import LM
import tensorflow as tf
import sys

'''
Reads the command line and receives its arguments from there:

Options
    -test <file>
        Test file as a text file.
    -restore <file>
        File of the previously trained model

'''

def read_file(file_name):
    batch_size = 10
    n_hidden = 512
    embedding = 300
    all_words = True
    train_file = None
    appearances = 3
    voc_file = None
    time_steps = 10
    with open(file_name, 'r') as file:
        line = file.readline()
        while line:
            content = line.split()
            if content == []:
                break
            elif content[0] == "BATCH_SIZE":
                batch_size = int(content[2])
            elif content[0] == "TIME_STEPS":
                time_steps = int(content[2])
            elif content[0] == "N_HIDDEN":
                n_hidden = int(content[2])
            elif content[0] == "EMBED_SIZE":
                embedding = int(content[2])
            elif content[0] == "ALL_WORDS":
                if content[2] == "False":
                    all_words = False
                elif content[2] == "True":
                    all_words = True
            elif content[0] == "APPEARANCES":
                appearances = int(content[2])
            elif content[0] == "TRAIN_FILE":
                train_file = content[2]
            elif content[0] == "VOC_FILE":
                if content[2] == "None":
                    voc_file = None
                else:
                    voc_file = content[2]
            line = file.readline()

    return train_file, batch_size, n_hidden, embedding, all_words, appearances, voc_file, time_steps

def print_help():

    print()
    print("Options:")
    print("     -test <file>")
    print("         Test file as a text file.")
    print("     -saved_model_dir <file>")
    print("         File of the previously trained model")
    print()


def read_cmd(args):

    test_file = None
    saved_model_dir = "lm_model.ckpt"

    # At position 0, we have LMcmd.py, so we start from 1
    iter = 1
    while iter < len(args):
        if args[iter] == "-test":
            test_file = args[iter+1]
        elif args[iter] == "-saved_model_dir":
            saved_model_dir = args[iter+1]
        elif args[iter] == "-restore":
            saved_model_dir = args[iter + 1]

        iter += 2

    return test_file, saved_model_dir


if sys.argv[1] == "help" or sys.argv[1] == "HELP" or sys.argv[1] == "-help":
    print_help()

else:
    print("Reading the command line..")
    test_file, saved_model_dir = read_cmd(sys.argv)
    print("Finished reading the command line!")

    print("Reading the file..")
    train_file, batch_size, n_hidden, embed_size, all_words, appearances, voc_file, time_steps = read_file("lm_cmd.txt")
    print("Finished reading the file!")

    print("Setting up the LM..")
    LSTM_cell = LM(training_file=train_file, n_epochs= 40, time_steps=time_steps, batch_size=batch_size,
                   n_hidden=n_hidden, embed_size=embed_size, all_words=all_words, min_nb_appearances=appearances,
                    valid_file=test_file, voc_file=voc_file)

    print("Voc size: ", LSTM_cell.voc_size)
    print("Text size: ", LSTM_cell.text_size)

    print("Starting the session..")
    with tf.Session() as sess:
        print("Started testing..")
        LSTM_cell.test(sess, restore_dir=saved_model_dir)

