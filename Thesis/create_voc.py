import numpy as np
import collections as coll
import io

nb = [i for i in range(1, 100)]

dir = "steven/data/data/"
min_appearances = 250
max_min_appearances = 600
unknown_string = "<UNK>"

temp_dict = {}


for i in range(len(nb)):
    if nb[i] >= 10:
        file_name = "news.en-000" + str(nb[i]) + "-of-00100"
    else:
        file_name = "news.en-0000" + str(nb[i]) + "-of-00100"

    print("Now file: ", file_name)
    with io.open(dir + file_name, "r", encoding="utf8") as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)


    print("Text size: ", len(content))

    count = coll.Counter(content).most_common()

    print("Counted them!")

    print("Different words found: ", len(count))


    print("Now updating temp_dict..")

    for word, word_count in count:
        if word in temp_dict.keys():
            val = temp_dict[word]
            temp_dict[word] = val + word_count
        else:
            temp_dict[word] = word_count

    print("Voc size so far without removing rare words: ", len(temp_dict))

appearances = min_appearances
while appearances < max_min_appearances:
    dictionary = {}
    for word in temp_dict.keys():
        if temp_dict[word] >= appearances:
            dictionary[word] = len(dictionary)

    if unknown_string not in dictionary.keys():
        dictionary[unknown_string] = len(dictionary)

    print("Appearances: ", appearances)
    print("Voc size: ", len(dictionary))
    with open("steven/data/voc" + str(appearances) + ".txt", "w", encoding="utf8") as f:
        pass
        for word in dictionary.keys():
            f.write(word + " ")

    appearances += 25
    print()




