#! /usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import math
import random
from scipy import spatial
import LM_PrintFunctions
import time
import sklearn.neighbors as sk

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                SECTION 1: RETURNING INPUT AND OUTPUT BATCHES
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# Returns the input and output batches, updates the offsets and epochs
        # Input: a list of offsets, either empty or of [batch_size]
        # Input: the epoch number
        # Input: the number of input words
        # Output: a matrix symbols_in_keys of dimension [batch_size, n_input]
        # Output: a matrix symbols_out_onehot of dimension [n_input, batch_size, voc_size]
        # Output: the updated list of offsets new_offsets, ready for the next step
        # Output: the old list of offsets
        # Output: the updated epoch number n_epoch
        # Output: boolean epoch_update indicating whether n_epoch was updated
        # Output: sequence length
'''
Returns the next input and output batch, updates the offsets and epochs

The training_data is divided into batch_size more or less equal subtexts. Offsets are initialized in this way.
'''
def next_batch(offsets, n_epoch, n_input, batch_size, training_data, voc_size, text_size, dictionary, unknown_string):

    sequence_range = text_size // batch_size

    if offsets == []:
        offsets = [i * sequence_range for i in range(batch_size)]


    symbols_in_keys = []
    for j in range(batch_size):
        temp = []
        for i in range(offsets[j], offsets[j] + n_input):
            try:
                temp.append(dictionary[training_data[i]])
            except:
                temp.append(dictionary[unknown_string])
        symbols_in_keys.append(temp)


    # symbols_in_keys = [[dictionary[training_data[i]] for i in range(offsets[j], offsets[j] + n_input)]
    #                           for j in range(batch_size)]

    symbols_out_onehot = np.zeros([n_input, batch_size, voc_size], dtype=float)
    for i in range(batch_size):
        for j in range(n_input):
            try:
                symbols_out_onehot[j][i][dictionary[training_data[offsets[i] + j + 1]]] = 1.0
            except:
                symbols_out_onehot[j][i][dictionary[unknown_string]] = 1.0
    symbols_out_onehot = np.reshape(symbols_out_onehot, [n_input, batch_size, voc_size])

    new_offsets = [offsets[i] for i in range(batch_size)]

    epoch_update = False

    for i in range(batch_size):
        new_offsets[i] += n_input
        # if new_offsets[i] + n_input > sequence_range * (i + 1):
        #     epoch_update = True
        if new_offsets[i] + n_input > text_size:
            epoch_update = True

    if epoch_update:
        new_offsets = [i * sequence_range for i in range(batch_size)]
        n_epoch += 1


    return symbols_in_keys, symbols_out_onehot, new_offsets, offsets, n_epoch, epoch_update

# Returns the input and output batches, updates the offsets and epochs
        # Input: a list of offsets, either empty or of [batch_size]
        # Input: the epoch number
        # Input: the number of input words
        # Output: a matrix symbols_in_keys of dimension [batch_size, n_input]
        # Output: a matrix symbols_out_onehot of dimension [n_input, batch_size, voc_size]
        # Output: the updated list of offsets new_offsets, ready for the next step
        # Output: the old list of offsets
        # Output: the updated epoch number n_epoch
        # Output: boolean epoch_update indicating whether n_epoch was updated
        # Output: sequence length
'''
Returns the next input and output batch, updates the offsets and epochs

Offsets are initialized randomly.
'''
def next_batch_rand(offsets, n_epoch, n_input, batch_size, training_data, voc_size, text_size, dictionary, step):

    if offsets == []:
        offsets = [random.randint(0, text_size - n_input - 1) for i in range(batch_size)]

    symbols_in_keys = [[dictionary[str(training_data[i])] for i in range(offsets[j], offsets[j] + n_input)]
                       for j in range(batch_size)]

    symbols_out_onehot = np.zeros([n_input, batch_size, voc_size], dtype=float)
    for i in range(batch_size):
        for j in range(n_input):
            symbols_out_onehot[j][i][dictionary[str(training_data[offsets[i] + j + 1])]] = 1.0
    symbols_out_onehot = np.reshape(symbols_out_onehot, [n_input, batch_size, voc_size])

    new_offsets = [offsets[i] + 1 for i in range(batch_size)]

    epoch_update = False

    for i in range(batch_size):
        if new_offsets[i] + n_input + 1 > text_size:
            new_offsets[i] = random.randint(0, text_size - n_input - 1)

    if step == text_size // batch_size:
        epoch_update = True
        n_epoch += 1
        new_offsets = [random.randint(0, text_size - n_input - 1) for i in range(batch_size)]



    return symbols_in_keys, symbols_out_onehot, new_offsets, offsets, n_epoch, epoch_update


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                SECTION 2: UPDATING THE DICTIONARIES
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'''
Updates the dictionaries of the states h and c. For each word, the state is the average of the previously seen states.
'''
def update_states(states_c_dict, states_h_dict, word_count, final_state, batch_size, data, offsets):

    for i in range(batch_size):
        word = data[offsets[i]]
        if word in word_count:
            n = word_count[word]
            states_c_dict = update_avg_state(states_c_dict, n, word, final_state.c[i])
            states_h_dict = update_avg_state(states_h_dict, n, word, final_state.h[i])
            word_count[word] = n + 1
        else:
            word_count[word] = 1
            states_h_dict[word] = final_state.h[i]
            states_c_dict[word] = final_state.c[i]

    return states_c_dict, states_h_dict, word_count

'''
Updates the dictionary of the word embedding, which serves as the input for the LM
'''
def update_embbeding(word_embedding, word_count, embedding, batch_size, data, offsets):

    for i in range(batch_size):
        word = data[offsets[i]]
        if word in word_count:
            n = word_count[word]
            word_embedding = update_avg_state(word_embedding, n, word, embedding[i])
            word_count[word] = n + 1
        else:
            word_count[word] = 1
            word_embedding[word] = embedding[i]

    return word_embedding, word_count


def update_avg_state(states_dict, n, word, last_state):

    new_state = states_dict[word] * n + last_state
    new_state = new_state / (n + 1)


    states_dict[word] = new_state

    return states_dict

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                SECTION 3: ANSWERING THE QUESTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

'''
Answer the questions of the qst_file given a states_c_dict and possibly states_h_dict which both represent
a word embedding. 

distance must be: 
    - euclidean
    - norm_euclidean
    - manhattan
    - inf
    - cosine
'''
def ask_questions(qst_file, states_c_dict, states_h_dict = None, distance = "norm_euclidean", k_neighbors = 1,
                                                                                multi_vector = False):

    print("STARTING THE QUESTIONS")
    # Parameters that depend on the distance
    vp, normalize, p = distance_parameters(distance)

    # Based on states_h_dict, we determine whether we need to check two states or not
    if states_h_dict is None:
        two_states = False
    else:
        two_states = True

    # Depth and number of k-d trees
    nb_trees = 2
    depth = 400
    print("nb_trees: ", nb_trees)
    print("depth: ", depth)

    print_q = 50


    # Handling the one state case
    if two_states is False:
        trees, answers = built_trees(states_c_dict, nb_trees=nb_trees, depth=depth, normalize=normalize, vp=vp,
                                     k_neighbors=k_neighbors)
        print("kd-trees are built!")

        # Initialize
        c_answers = {True: 0, False: 0, None: 0}

        # Reading the questions
        with open(qst_file, "r") as file:
            line = file.readline()
            # Timer to keep track of the average time per question
            start = time.time()
            k = 0
            q = 1
            while line:
                content = line.split()
                # If we encounter a new category (title), we print and we re-initialize
                if content[0] == ":":
                    if k > 0:
                        print("Questions answered: ", q)
                        stop = time.time()
                        print("Time: ", stop - start)
                        print("Average time per question: ", (stop - start) / (q * 1.0))
                        start = time.time()
                    c_answers = initialize(line, c_answers)
                    q = 1
                    k = k + 1
                # If not, answer the question
                else:
                    # Answer the question, given the question, the states_dict, the trees and the corresponding answers
                    c_answer = question(content, states_c_dict, trees=trees, answers=answers, p=p, vp=vp,
                                        k_neighbors=k_neighbors, multi_vector=multi_vector)
                    if q % print_q == 0 and q != 0:
                        print("Answering question: ", q)
                    c_answers[c_answer] += 1
                    q = q + 1
                line = file.readline()
        # Print the results for the last category
        print("Questions answered: ", (q - 1))
        stop = time.time()
        print("Time: ", stop - start)
        print("Average time per question: ", (stop - start) / (q * 1.0))
        print("State: ")
        print(c_answers)


    # Same as before, but this time everything is performed twice, for the two states.
    else:
        trees_h, answers_h = built_trees(states_h_dict, nb_trees=nb_trees, depth=depth, normalize=normalize, vp=vp,
                                         k_neighbors=k_neighbors)
        trees_c, answers_c = built_trees(states_c_dict, nb_trees=nb_trees, depth=depth, normalize=normalize, vp=vp,
                                         k_neighbors=k_neighbors)

        print("kd-trees are built!")

        c_answers = {True: 0, False: 0, None: 0}
        h_answers = {True: 0, False: 0, None: 0}

        with open(qst_file, "r") as file:
            line = file.readline()
            start = time.time()
            k = 0
            q = 1
            while line:
                content = line.split()
                if content[0] == ":":
                    if k > 0:
                        print("Questions answered: ", q)
                        stop = time.time()
                        print("Time: ", stop - start)
                        print("Average time per question: ", (stop - start) / (q * 2.0))
                        start = time.time()
                    c_answers, h_answers = initialize(line, c_answers, h_answers)
                    q = 1
                    k = k + 1
                else:
                    c_answer = question(content, states_c_dict, trees=trees_c, answers=answers_c, p=p, vp=vp,
                                        k_neighbors=k_neighbors, multi_vector=multi_vector)
                    h_answer = question(content, states_h_dict, trees=trees_h, answers=answers_h, p=p, vp=vp,
                                        k_neighbors=k_neighbors, multi_vector=multi_vector)
                    if q % print_q == 0 and q != 0:
                        print("Answering question: ", q)
                    c_answers[c_answer] += 1
                    h_answers[h_answer] += 1
                    q = q + 1
                line = file.readline()
        print("Questions answered: ", (q - 1))
        stop = time.time()
        print("Time: ", stop - start)
        print("Average time per question: ", (stop - start) / (q * 2.0))
        print("Cell state: ")
        print(c_answers)
        print("Hidden state: ")
        print(h_answers)


'''
Answer the given question. The vectors are provided by the states_dict
'''
def question(content, states_dict, trees = None, answers = None, p = 2, vp = False, k_neighbors = 1, multi_vector = False):

    for i in range(len(content)):
        if content[i] not in states_dict.keys():
            return None

    res = states_dict[content[0]] - states_dict[content[1]] + states_dict[content[3]]

    if vp is False:
        # in this case, a query performed on the kd_trees returns the distance as well as the index
        # of the vector with the shortest distance
        near_dist_list = []
        index_list = []
        for i in range(len(trees)):
            (near_dist, index) = trees[i].query(res, p=p, k=k_neighbors)
            near_dist_list, index_list = update(near_dist_list, index_list, near_dist, index, k_neighbors=k_neighbors)
        tree_indices, local_indices = trees_and_local_indices(near_dist_list, k_neighbors=k_neighbors)

        return answer_question(answers, index_list, tree_indices, local_indices, content[2], k_neighbors=k_neighbors,
                               multi_vector=multi_vector)
    else:
        near_dist_list = []
        index_list = []
        for i in range(len(trees)):
            (near_dist, index) = trees[i].kneighbors([res])
            near_dist_list, index_list = update(near_dist_list, index_list, near_dist[0], index[0],
                                                k_neighbors=k_neighbors, vp=vp)
        tree_indices, local_indices = trees_and_local_indices(near_dist_list, k_neighbors=k_neighbors)

        return answer_question(answers, index_list, tree_indices, local_indices, content[2], k_neighbors=k_neighbors,
                               multi_vector=multi_vector)


'''
Return the trees list and answers list, given the states dictionary and the preferred number of trees
'''
def built_trees(states_dict, nb_trees, depth, normalize, vp, k_neighbors = 1):

    pp_tree = len(states_dict) // nb_trees
    trees = []
    answers = []
    for i in range(nb_trees):
        if i < nb_trees - 1:
            tree, answer = get_tree_from_states(states_dict, start=i * pp_tree + 1, stop=(i + 1) * pp_tree,
                                        depth=depth, normalize=normalize, vp=vp, k_neighbors=k_neighbors)
        else:
            tree, answer = get_tree_from_states(states_dict, start=i * pp_tree + 1, stop=len(states_dict),
                                        depth=depth, normalize=normalize, vp=vp, k_neighbors=k_neighbors)
        trees.append(tree)
        answers.append(answer)

    return trees, answers

'''
Update the near_dist_list and index_list based on k_neighbors 
'''
def update(near_dist_list, index_list, near_dist, index, k_neighbors = 1, vp = False):

    if k_neighbors == 1 and vp is False:
        near_dist_list.append(near_dist)
        index_list.append(index)
    else:
        near_dist_list.extend(near_dist)
        index_list.extend(index)

    return near_dist_list, index_list


'''
Returns the local indices as well as the indices of the trees, so that the k_neighbors nearest neighbors can be found
'''
def trees_and_local_indices(near_dist_list, k_neighbors = 1):

    tree_indices = []
    local_indices = []

    for i in range(k_neighbors):
        index = near_dist_list.index(min(near_dist_list))
        tree_indices.append(index // k_neighbors)
        local_indices.append(index)
        near_dist_list.pop(index)

    return tree_indices, local_indices


'''
Returns whether the question was answered or not based on the k_neighbors nearest neighbors
'''
def answer_question(answers, index_list, tree_indices, local_indices, real_answer, k_neighbors = 1, multi_vector = False):

    if multi_vector is False:
        for i in range(k_neighbors):
            if answers[tree_indices[i]][index_list[local_indices[i]]] == real_answer:
                return True
        return False
    else:
        for i in range(k_neighbors):
            if answers[tree_indices[i]][index_list[local_indices[i]]][0] == real_answer:
                return True
        return False


'''
 Initialize the answer dictionary, as well as to print results once we come across a new category
'''
def initialize(line, c_answers, h_answers = None):

    if h_answers is None:
        # This function is called in the beginning, and each time we encounter a new category.
        # In the latter case, we should also print the results from the previous category.
        if c_answers[True] + c_answers[False] + c_answers[None] > 0:
            print("State: ")
            print(c_answers)

        # Initialize
        c_answers = {True: 0, False: 0, None: 0}

        print()
        print()
        print("Category" + line)

        return c_answers

    else:
        if c_answers[True] + c_answers[False] + c_answers[None] > 0:
            print("Cell state: ")
            print(c_answers)
            print("Hidden state: ")
            print(h_answers)
        c_answers = {True: 0, False: 0, None: 0}
        h_answers = {True: 0, False: 0, None: 0}

        print()
        print()
        print("Category" + line)

        return c_answers, h_answers



'''
Given the states dict and a start and stop index, as well as a depth, it builds a tree of the given depth
based on the elements of the states_dict between start and stop+1
'''
def get_tree_from_states(states_dict, start = None, stop = None, depth = 10, normalize = False, vp = False,
                         k_neighbors = 1):

    if start is None or stop is None:
        start = 0
        stop = len(states_dict) - 1

    if normalize is True:
        for word in states_dict.keys():
            nm_vector = normalize_vector(states_dict[word])
            states_dict[word] = nm_vector

    answer = list(states_dict.keys())
    answer = answer[start:stop+1]
    points = list(states_dict.values())
    points = points[start:stop+1]

    if vp is False:
        kd_tree = spatial.KDTree(points, depth)
        return kd_tree, answer
    else:
        ball_tree = sk.NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        ball_tree.fit(points)
        return ball_tree, answer


'''
Normalize a vector
'''
def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

'''
Returns the parameters needed in the process of answering questions based on the preferred distance
'''
def distance_parameters(distance):

    p = 2
    normalize = False
    ball = False

    if distance == "norm_euclidean":
        normalize = True
    elif distance == "cosine":
        ball = True
    elif distance == "manhattan":
        p = 1
    elif distance == "inf":
        p = math.inf

    return ball, normalize, p



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                SECTION 4: OTHER FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

'''
Write the given text to the file with the given name, by appending it at the end of the file.
'''
def write_to_file(string, file):

    with open(file, "a+") as f:
        f.writelines(string  + "\n")



