import numpy as np
from LM_HelpFunctions import ask_questions
import sys

"""
Reads the command line and prints the performance on the questions, given the input word embedding.

Options
        -word_embed <file>
            File of the word embedding
        -questions <file>
            File of the questions
"""

def print_help():
    print()
    print("Options:")
    print("     -word_embed <file>")
    print("         File of the word embedding")
    print("     -questions <file>")
    print("         File of the questions")
    print("     -distance <int>")
    print("         0 for Euclidean distance")
    print("         1 for normalized Euclidean")
    print("         2 for Manhattan distance")
    print("         3 for infinite distance")
    print("         4 for cosine distance")
    print("             - default is 1")
    print("     -k <int>")
    print("         Number of neighbors, default is 1")
    print()


def read_cmd(args):
    word_embed_file = "word2vec-master/output.txt"
    questions_file = "questions.txt"
    dist_int = 1
    k_neighbors = 1

    iter = 1
    while iter < len(args):
        if args[iter] == "-word_embed":
            word_embed_file = args[iter+1]
        elif args[iter] == "-questions":
            questions_file = args[iter+1]
        elif args[iter] == "-distance":
            dist_int = int(args[iter+1])
        elif args[iter] == "-k":
            k_neighbors = int(args[iter+1])

        iter += 2

    distance = "norm_euclidean"

    if dist_int == 0:
        distance = "euclidean"
    elif dist_int == 2:
        distance = "manhattan"
    elif dist_int == 3:
        distance = "inf"
    elif dist_int == 4:
        distance = "cosine"


    return word_embed_file, questions_file, distance, k_neighbors



def construct_dict(file_name):

    word_embeddings = {}

    def extract_vector(content, N):

        vector = np.zeros(N)
        for i in range(len(content)):
            vector[i] = float(content[i])
        return vector

    with open(file_name, "r") as file:
        N = int(file.readline().split()[1])
        line = file.readline()
        while line:
            content = line.split()
            word = content[0]
            vector = extract_vector(content[1:len(content)], N)
            word_embeddings[word] = vector
            line = file.readline()

    return word_embeddings


if len(sys.argv) > 1 and (sys.argv[1] == "help" or sys.argv[1] == "HELP" or sys.argv[1] == "-help"):
    print_help()

else:
    word_embed_file, questions_file, distance, k_neighbors = read_cmd(sys.argv)
    word_embedding = construct_dict(word_embed_file)
    print("Voc size: ", len(word_embedding))
    ask_questions(questions_file, word_embedding, distance=distance, k_neighbors=k_neighbors)

