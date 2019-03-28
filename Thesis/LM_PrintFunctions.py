#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import math
import sys
from scipy import spatial

def print_output_RNN(step, display_step, loss_total, acc_total, old_offsets, onehot_pred, batch_size,
                     n_input, training_data, reverse_dictionary):
    if (step + 1) % display_step == 0:
        print("Iter =" + str(step + 1), ", Average loss= " +
              "{:.6f}".format(loss_total / display_step) + ", Perplexity= " +
              "{:.6f}".format(math.exp(loss_total / display_step)) + ", Average accuracy= " +
              "{:.2f}".format(100 * acc_total / display_step))
        acc_total = 0
        loss_total = 0
        symbols_in, symbols_out, symbols_out_pred = None, None, None
        for j in range(batch_size):
            symbols_in = [training_data[i] for i in range(old_offsets[j], old_offsets[j] + n_input)]
            symbols_out = training_data[old_offsets[j] + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred[j], 0).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
    return acc_total, loss_total

def print_output_RNN_all(step, display_step, loss_total, acc_total, old_offsets, onehot_pred, batch_size,
                     n_input, training_data, reverse_dictionary):
    if (step + 1) % display_step == 0:
        print("Iter =" + str(step + 1), ", Average loss= " +
              "{:.6f}".format(loss_total / display_step) + ", Perplexity= " +
              "{:.6f}".format(math.exp(loss_total / display_step)) + ", Average accuracy= " +
              "{:.2f}".format(100 * acc_total / display_step))
        acc_total = 0
        loss_total = 0
        symbols_in, symbols_out, symbols_out_pred = None, None, None
        for j in range(batch_size):
            symbols_in = [training_data[i] for i in range(old_offsets[j], old_offsets[j] + n_input)]
            symbols_out = [training_data[old_offsets[j] + i] for i in range(1, n_input + 1)]
            symbols_out_pred = [reverse_dictionary[int(tf.argmax(onehot_pred[i][j], 0).eval())] for i in range(n_input)]
            print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
    return acc_total, loss_total

def print_output_RNN_epoch(step, loss_total, acc_total, epoch, file = "output_lm_model.txt"):
    print("Epoch = " + str(epoch), ", Average loss = " +
              "{:.6f}".format(loss_total / step) + ", Perplexity = " +
              "{:.6f}".format(math.exp(loss_total / step)) + ", Average accuracy = " +
              "{:.2f}".format(100 * acc_total / step))

    with open(file, "a+") as file:
        file.writelines("Epoch = " + str(epoch) + ", Average loss = " +
              "{:.6f}".format(loss_total / step) + ", Perplexity = " +
              "{:.6f}".format(math.exp(loss_total / step)) + ", Average accuracy = " +
              "{:.2f}".format(100 * acc_total / step))

    acc_total = 0
    loss_total = 0
    symbols_in, symbols_out, symbols_out_pred = None, None, None
    # if print_predicted_words:
    #     for j in range(batch_size):
    #         symbols_in = [training_data[i] for i in range(old_offsets[j], old_offsets[j] + n_input)]
    #         symbols_out = [training_data[old_offsets[j] + i] for i in range(1, n_input + 1)]
    #         symbols_out_pred = [reverse_dictionary[int(tf.argmax(onehot_pred[i][j], 0).eval())] for i in range(n_input)]
    #         print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
    return acc_total, loss_total

def print_predicted_words(batch_size, training_data, old_offsets, n_input, reverse_dictionary, onehot_pred):
    for j in range(batch_size):
        symbols_in = [training_data[i] for i in range(old_offsets[j], old_offsets[j] + n_input)]
        symbols_out = [training_data[old_offsets[j] + i] for i in range(1, n_input + 1)]
        symbols_out_pred = [reverse_dictionary[int(tf.argmax(onehot_pred[i][j], 0).eval())] for i in range(n_input)]
        print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
    print()



def print_state_distance(selected_word, states_dict, reverse_dictionary, write_to_file = False,
                         file_name = "state_h.txt", inorout_string = "", c_or_h_state = "h"):
    if selected_word not in states_dict.keys():
        return
    state_sel_word = states_dict[selected_word]
    min_distance_e, min_distance_cos = sys.maxsize, sys.maxsize
    closest_word_e, closest_word_cos = None, None

    for i in reverse_dictionary.keys():
        word = reverse_dictionary[i]
        if word in states_dict.keys():
            state_iter_word = states_dict[word]
            current_distance_e = np.linalg.norm(state_sel_word - state_iter_word)
            current_distance_cos = spatial.distance.cosine(state_sel_word, state_iter_word)
            if (current_distance_e < min_distance_e and word != selected_word):
                min_distance_e = current_distance_e
                closest_word_e = word
            if (current_distance_cos < min_distance_cos and word != selected_word):
                min_distance_cos = current_distance_cos
                closest_word_cos = word

    print("State " + c_or_h_state + " " + inorout_string +  ": closest word to ", selected_word, ": ", closest_word_e, " - distance: ", min_distance_e)
    print("State "+ c_or_h_state + " " + inorout_string + ": closest word to ", selected_word, ": ", closest_word_cos, " - distance: ", min_distance_cos)

    if write_to_file is True:
        with open(file_name, "a") as file:
            file.write("State_h "+ inorout_string + ": closest word to " + str(selected_word) + ": " + str(closest_word_e)
                       + " - distance: " + str(min_distance_e) + "\n")
            file.write("State_h " + inorout_string + ": closest word to " + str(selected_word) + ": " + str(closest_word_cos)
                       + " - distance: " + str(min_distance_cos) + "\n")

def print_distance_to_word(selected_word, states_dict, reverse_dictionary, write_to_file = False,
                         file_name = "state_h.txt", inorout_string = "", c_or_h_state = "h"):
    if selected_word not in states_dict.keys():
        return
    state_sel_word = states_dict[selected_word]

    for i in reverse_dictionary.keys():
        word = reverse_dictionary[i]
        if word in states_dict.keys():
            state_iter_word = states_dict[word]
            current_distance_e = np.linalg.norm(state_sel_word - state_iter_word)
            current_distance_cos = spatial.distance.cosine(state_sel_word, state_iter_word)
            print("State " + c_or_h_state + " " + inorout_string + ": closest word to ", selected_word, ": ",
                  word, " - distance: ", current_distance_e)
            print("State " + c_or_h_state + " " + inorout_string + ": closest word to ", selected_word, ": ",
                  word, " - distance: ", current_distance_cos)

            if write_to_file is True:
                with open(file_name, "a") as file:
                    file.write("State_h "+ inorout_string + ": closest word to " + str(selected_word) + ": " + str(word)
                               + " - distance: " + str(current_distance_e) + "\n")
                    file.write("State_h " + inorout_string + ": closest word to " + str(selected_word) + ": " + str(word)
                               + " - distance: " + str(current_distance_cos) + "\n")



def print_time(time):
    minutes = time // 60
    seconds = time % 60
    hours = minutes // 60
    minutes = minutes % 60
    if (hours == 0):
        print("Time: " + str(int(minutes)) + " minutes " + str(int(seconds)) + " seconds")
    else:
        print("Time: " + str(int(hours)) + " hours ", str(int(minutes)) + " minutes " + str(int(seconds)) + " seconds")

def print_final_stats(loss_total, acc_total, iter_total):
    print()
    print("Training finished!")
    print("Total number of iterations: ", iter_total)

def print_acc_loss_ppl(loss_total, acc_total, step):
    print("Average loss = " + "{:.6f}".format(loss_total / step) + ", Perplexity = " +
          "{:.6f}".format(math.exp(loss_total / step)) + ", Average accuracy = " +
          "{:.2f}".format(100 * acc_total / step))

    with open("output_lm_model.txt", "w") as file:
        file.writelines("Average loss = " +
                   "{:.6f}".format(loss_total / step) + ", Perplexity = " +
                   "{:.6f}".format(math.exp(loss_total / step)) + ", Average accuracy = " +
                   "{:.2f}".format(100 * acc_total / step) + "\n")


# Print the result of the questions answered
def print_questions_result(dict):
    tot_qst_answ = dict[True] + dict[False]

    print("Answered: ", tot_qst_answ / (tot_qst_answ + dict[None]) * 100)
    print("Unanswered: ", dict[None] / (tot_qst_answ + dict[None]) * 100)
    if tot_qst_answ > 0:
        print("Of the questions answered:")
        print("Correct: ", dict[True] / tot_qst_answ * 100)
        print("Wrong: ", dict[False] / tot_qst_answ * 100)

