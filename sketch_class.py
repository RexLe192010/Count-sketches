import numpy as np
from hashlib import md5
import mmh3
import random
import sys
import heapq

# Count-Min Sketch
class CountMinSketch:
    def __init__(self, R):
        self.width = R  # the width of the table
        self.depth = 5  # the depth of the table, which is also the number of hash functions
        self.table = np.zeros((self.depth, self.width), dtype=int) # the table, consisting of depth * width slots
        # Use numpy's new generator, which supports larger ranges
        rng = np.random.default_rng()
        self.hash_seeds = rng.integers(0, 2**32 - 1, size=self.depth, dtype=np.uint64)
        self.top_k_heap = TopKHeap(500)  # a heap to store the top 500 frequent items

    def murmurhash(self, item, i):
        # murmurhash function
        seed = self.hash_seeds[i].item()
        return mmh3.hash(str(item), seed) % self.width

    def insert(self, item):
        # insert function: add 1 to each position of the hash function
        for i in range(self.depth):
            self.table[i, self.murmurhash(item, i)] += 1
        
        # update the top k heap
        count = self.query(item)
        self.top_k_heap.add_or_update(item, count)

    def query(self, item):
        # query function: return the minimum value of the hash function
        return min(self.table[i, self.murmurhash(item, i)] for i in range(self.depth))
    
    def get_top_k(self, k):
        # return the top k frequent items and their counts
        return self.top_k_heap.get_top_k()

# Count-Median Sketch
class CountMedianSketch:
    def __init__(self, R):
        self.width = R
        self.depth = 5
        self.table = np.zeros((self.depth, self.width), dtype=int)
        # Use numpy's new generator, which supports larger ranges
        rng = np.random.default_rng()
        self.hash_seeds = rng.integers(0, 2**32 - 1, size=self.depth, dtype=np.uint64)
        self.top_k_heap = TopKHeap(500)  # a heap to store the top 500 frequent items
    
    def murmurhash(self, item, i):
        # murmurhash function
        seed = self.hash_seeds[i].item()
        return mmh3.hash(str(item), seed) % self.width

    def insert(self, item):
        for i in range(self.depth):
            self.table[i, self.murmurhash(item, i)] += 1
        
        # update the top k heap
        count = self.query(item)
        self.top_k_heap.add_or_update(item, count)

    def query(self, item):
        return np.median([self.table[i, self.murmurhash(item, i)] for i in range(self.depth)])
    
    def get_top_k(self, k):
        # return the top k frequent items and their counts
        return self.top_k_heap.get_top_k()

# Count Sketch
class CountSketch:
    def __init__(self, R):
        self.width = R
        self.depth = 5
        self.table = np.zeros((self.depth, self.width), dtype=int)
        # Use numpy's new generator, which supports larger ranges
        rng = np.random.default_rng()
        self.hash_seeds = rng.integers(0, 2**32 - 1, size=self.depth, dtype=np.uint64)
        self.top_k_heap = TopKHeap(500)  # a heap to store the top 500 frequent items

    def murmurhash(self, item, i):
        # murmurhash function
        seed = self.hash_seeds[i].item()
        return mmh3.hash(str(item), seed) % self.width

    def _sign_hash(self, item, i):
        np.random.seed(self.hash_seeds[i])
        return np.random.choice([-1, 1])

    def insert(self, item):
        for i in range(self.depth):
            h = self.murmurhash(item, i)
            sign = self._sign_hash(item, i)
            self.table[i, h] += sign

        # update the top k heap
        count = self.query(item)
        self.top_k_heap.add_or_update(item, count)

    def query(self, item):
        return np.median([self._sign_hash(item, i) * self.table[i, self.murmurhash(item, i)] for i in range(self.depth)])
    
    def get_top_k(self, k):
        # return the top k frequent items and their counts
        return self.top_k_heap.get_top_k()


# Exact Count
class ExactCount:
    def __init__(self):
        self.counts = {}

    def insert(self, item):
        # insert a word into the dictionary
        if item in self.counts:
            self.counts[item] += 1
        else:
            self.counts[item] = 1

    def query(self, item):
        # return the count of a word in the dictionary
        return self.counts.get(item, 0)

    def total_unique_items(self):
        # return the number of unique items in the dictionary
        return len(self.counts)

    def get_top_k_frequent(self, k=100):
        # return the top k frequent items and their counts
        return sorted(self.counts.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_top_k_infrequent(self, k=100):
        # return the top k infrequent items and their counts
        return sorted(self.counts.items(), key=lambda x: x[1])[:k]

    def get_random_k_tokens(self, k=100):
        # return k random items and their
        return random.sample(list(self.counts.items()), k)

    def space_used(self):
        # return the space used by the dictionary
        return sys.getsizeof(self.counts)
    


class TopKHeap:
    def __init__(self, k):
        self.k = k
        self.heap = []
        self.entry_finder = {}  # map from token to its index in the heap

    def add_or_update(self, token, count):
        # insert or update a token in the heap
        if token in self.entry_finder:
            # update the count of the token
            index = self.entry_finder[token]
            self.heap[index] = (count, token)
            heapq.heapify(self.heap)  # re-heapify the heap
        else:
            if len(self.heap) < self.k:
                # the heap is not full yet
                heapq.heappush(self.heap, (count, token))
                self.entry_finder[token] = len(self.heap) - 1
            else:
                # if heap is full, we need to check if the new token should be added to the heap
                if count > self.heap[0][0]:
                    min_item = heapq.heappop(self.heap)
                    if min_item[1] in self.entry_finder:
                        del self.entry_finder[min_item[1]]
                    heapq.heappush(self.heap, (count, token))
                    self.entry_finder[token] = len(self.heap) - 1

    def get_top_k(self):
        # return the top k tokens in the heap
        return [(count, token) for count, token in sorted(self.heap, reverse=True)]
