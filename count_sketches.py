import numpy as np
from hashlib import md5

# Count-Min Sketch
class CountMinSketch:
    def __init__(self, R):
        self.width = R  # the width of the table
        self.depth = 5  # the depth of the table, which is also the number of hash functions
        self.table = np.zeros((self.depth, self.width), dtype=int)
        self.hash_seeds = np.random.randint(0, 2**32 - 1, size=self.depth)

    def _hash(self, item, i):
        # hash function
        return (int(md5((str(item) + str(self.hash_seeds[i])).encode('utf-8')).hexdigest(), 16) % self.width)

    def insert(self, item):
        # insert function: add 1 to each position of the hash function
        for i in range(self.depth):
            self.table[i, self._hash(item, i)] += 1

    def query(self, item):
        # query function: return the minimum value of the hash function
        return min(self.table[i, self._hash(item, i)] for i in range(self.depth))

# Count-Median Sketch
class CountMedianSketch:
    def __init__(self, R):
        self.width = R
        self.depth = 5
        self.table = np.zeros((self.depth, self.width), dtype=int)
        self.hash_seeds = np.random.randint(0, 2**32 - 1, size=self.depth)

    def _hash(self, item, i):
        return (int(md5((str(item) + str(self.hash_seeds[i])).encode('utf-8')).hexdigest(), 16) % self.width)

    def insert(self, item):
        for i in range(self.depth):
            self.table[i, self._hash(item, i)] += 1

    def query(self, item):
        return np.median([self.table[i, self._hash(item, i)] for i in range(self.depth)])

# Count Sketch
class CountSketch:
    def __init__(self, R):
        self.width = R
        self.depth = 5
        self.table = np.zeros((self.depth, self.width), dtype=int)
        self.hash_seeds = np.random.randint(0, 2**32 - 1, size=self.depth)

    def _hash(self, item, i):
        return (int(md5((str(item) + str(self.hash_seeds[i])).encode('utf-8')).hexdigest(), 16) % self.width)

    def _sign_hash(self, item, i):
        np.random.seed(self.hash_seeds[i])
        return np.random.choice([-1, 1])

    def insert(self, item):
        for i in range(self.depth):
            h = self._hash(item, i)
            sign = self._sign_hash(item, i)
            self.table[i, h] += sign

    def query(self, item):
        return np.median([self._sign_hash(item, i) * self.table[i, self._hash(item, i)] for i in range(self.depth)])
