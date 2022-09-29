import hnswlib
import numpy as np
import time

dim = 16
num_elements = 100000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

# We split the data in two batches:
data1 = data[:num_elements // 2]
data2 = data[num_elements // 2:]

# Declaring index
p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

# Initing index
# max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
# during insertion of an element.
# The capacity can be increased by saving/loading the index, see below.
#
# ef_construction - controls index search speed/build speed tradeoff
#
# M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
# Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

p.init_index(max_elements=num_elements//2, ef_construction=100, M=16)

# Controlling the recall by setting ef:
# higher ef leads to better accuracy, but slower search
p.set_ef(10)

# Set number of threads used during batch search/construction
# By default using all available cores
p.set_num_threads(4)


print("Adding first batch of %d elements" % (len(data1)))
p.add_items(data1)

# Serializing and deleting the index:
index_path='first_half.bin'
print("Saving index to '%s'" % index_path)
p.save_index(index_path)


# Query the elements for themselves and measure recall:
labels, distances = [], []
efs1 = [2, 10, 5, 10, 3]
efs2 = [3, 10, 5, 10, 3]
for i in range(5):
    p.set_ef(efs1[i])
    l, d = p.knn_query(data[-5+i], k=1)
    labels.append(l[0])
    distances.append(d[0])
labels = np.array(labels)
distances = np.array(distances)

print(labels, distances)
labels2, distances2 = p.knn_query_new(data[-5:], ef=efs2, k=1)
print(labels2, distances2)
