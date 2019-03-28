import numpy as np
from scipy import spatial
import time
import vptree
import sklearn.neighbors as sk
import math

# N = 55000
# points = np.random.randn(N, 10)
#
# print(points[:1])
# print(points[0])
#
# query = [.5] * 10
# cosine = lambda x, y: spatial.distance.cosine(x, y)
#
# dist = math.inf
# index = 0
#
# begin = time.time()
# for i in range(N):
#     if cosine(points[i], query) < dist:
#         dist = cosine(points[i], query)
#         index = i
# print("Brute-force took: ", time.time() - begin)
#
# print("Index: ", index)
# print("Distance: ", dist)
# print("Point:", points[index])
#
#
# #tree = sk.BallTree(points, metric=cosine)
# tree = sk.NearestNeighbors(n_neighbors=1, metric='cosine')
# tree.fit(points)
# begin = time.time()
# (d, i) = tree.kneighbors([query])
# print("KNN took: ", time.time() - begin)
#
# print("Index: ", i)
# print("Distance: ", d)
# print("Point:", points[i])
#


a = (1, 2)

print(a[1])