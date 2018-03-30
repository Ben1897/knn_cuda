import numpy as np
import knn

# query = np.array([[1,2,3,4,5],[6,7,8,9,10]], dtype='float32')
query = np.array([[1,2,3,4,5],[1,3,5,7,9]], dtype='float32')

dist, ind = knn.knn(query, query, 3)

print dist
print ind

# c = 128

# for n in range(4):
#     query = np.random.rand(c, 1000).astype(np.float32)

#     reference = np.random.rand(c, 4000).astype(np.float32)

#     # Index is 1-based
#     dist, ind = knn.knn(query.reshape(c, -1),
#                         reference.reshape(c, -1), 2)

#     print ind
