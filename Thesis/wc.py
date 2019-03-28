import sys
import io
import numpy as np
import collections
'''
Simple word count of a file:
    -text <file>
        file for which the total number of words must be determined
'''
def read_cmd(args):
    text = "steven/data/voc3.txt"

    iter = 1
    while iter < len(args):
        if args[iter] == "-text":
            text = args[iter+1]

        iter += 2

    return text

def read_data(file_name, encoding = "utf8"):
    with io.open(file_name, encoding=encoding) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content


text = read_cmd(sys.argv)

text_data = read_data(text)

count = collections.Counter(text_data).most_common()

dictionary = {}
for word, word_count in count:
    dictionary[word] = len(dictionary)

print("Text size: ", len(text_data))
print("Voc size: ", len(dictionary))

