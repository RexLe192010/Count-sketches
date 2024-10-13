from sketch_class import CountMinSketch, CountMedianSketch, CountSketch, ExactCount
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import csv

# setup the experiment
R_values = [2**10, 2**14, 2**18]
exact_count = ExactCount()
words = []
with open("../user-ct-test-collection-01.txt", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        words.append(row[1])




# compute the intersection of the top 500 words in the sketch and the top 100 words in the dictionary
def calculate_intersection(top_500_sketch, top_100_dict):
    # return the number of words that are in both sets
    return len(set(top_500_sketch).intersection(set(top_100_dict)))


# Function to compute sketch errors for each R value
def compute_sketch_errors(R, words, n):
    # Initialize the sketches
    count_min_sketch = CountMinSketch(R)
    count_median_sketch = CountMedianSketch(R)
    count_sketch = CountSketch(R)

    for word in words:
        print(word)
        count_min_sketch.insert(word)
        count_median_sketch.insert(word)
        count_sketch.insert(word)

    # get the top n frequent words
    freq_100 = exact_count.get_top_k_frequent(k=100)
    rand_100 = exact_count.get_top_k_frequent(k=100)
    infreq_100 = exact_count.get_top_k_infrequent(k=100)

    word_sets = {'Freq-100': freq_100, 'Rand-100': rand_100, 'Infreq-100': infreq_100}

    results = []

    for set_name, word_set in word_sets.items():
        # iterate over the words in the word set and compute the estimated counts
        words, true_counts = zip(*word_set)
        estimated_counts_min = [count_min_sketch.query(word) for word in words]
        estimated_counts_median = [count_median_sketch.query(word) for word in words]
        estimated_counts_sketch = [count_sketch.query(word) for word in words]

        errors_min = [abs(estimated - true) / true for estimated, true in zip(estimated_counts_min, true_counts)]
        errors_median = [abs(estimated - true) / true for estimated, true in zip(estimated_counts_median, true_counts)]
        errors_sketch = [abs(estimated - true) / true for estimated, true in zip(estimated_counts_sketch, true_counts)]

        # sort the words by their true counts
        sorted_indices = np.argsort(true_counts)
        sorted_words = np.array(words)[sorted_indices]
        sorted_errors_min = np.array(errors_min)[sorted_indices]
        sorted_errors_median = np.array(errors_median)[sorted_indices]
        sorted_errors_sketch = np.array(errors_sketch)[sorted_indices]

        results.append((R, set_name, sorted_words, sorted_errors_min, sorted_errors_median, sorted_errors_sketch))

    return results

# the main function to input word sets and compute the counts of each sketch
def plot_errors(words, n=100):
    # Count the frequency of each word, store the each word and its frequency in a dictionary
    for word in words:
        exact_count.insert(word)

    # Use ThreadPoolExecutor to process each R value in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_sketch_errors, R, words, n) for R in R_values]
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            for R, set_name, sorted_words, sorted_errors_min, sorted_errors_median, sorted_errors_sketch in results:
                # plot the errors (3 plots in each iteration)
                # there will be 3 (R) * 3 (word sets) * 3 (sketches) = 27 plots in total
                plt.figure(figsize=(10, 6))
                plt.plot(sorted_words, sorted_errors_min, label="Count-Min Sketch", color='r')
                plt.plot(sorted_words, sorted_errors_median, label="Count-Median Sketch", color='g')
                plt.plot(sorted_words, sorted_errors_sketch, label="Count-Sketch", color='b')
                
                plt.xlabel('Words (sorted by frequency)')
                plt.ylabel('Error')
                plt.title(f'Error Plot for {set_name} with R={R}')
                plt.legend()
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.show()


# Function to process each R value
def process_R_value(R, words, top_100_dict):
    # initialize sketches
    count_min_sketch = CountMinSketch(R)
    count_median_sketch = CountMedianSketch(R)
    count_sketch = CountSketch(R)

    # insert words into sketches
    for word in words:
        print(word)
        count_min_sketch.insert(word)
        count_median_sketch.insert(word)
        count_sketch.insert(word)

    # get the top 500 words from each sketch
    top_500_min = [token for _, token in count_min_sketch.get_top_k(k=500)]
    top_500_median = [token for _, token in count_median_sketch.get_top_k(k=500)]
    top_500_sketch = [token for _, token in count_sketch.get_top_k(k=500)]

    # calculate the intersection of the top 500 words in the sketch and the top 100 words in the dictionary
    intersection_min = calculate_intersection(top_500_min, top_100_dict)
    intersection_median = calculate_intersection(top_500_median, top_100_dict)
    intersection_sketch = calculate_intersection(top_500_sketch, top_100_dict)

    return (R, intersection_min, intersection_median, intersection_sketch)

# plot the intersection of the top 500 words in the sketch and the top 100 words in the dictionary
def plot_top_500_intersection(words):
    intersections_min = []
    intersections_median = []
    intersections_sketch = []

    # get the top 100 words from the dictionary
    top_100_dict = [token for token, _ in exact_count.get_top_k_frequent(100)]

    # Use ThreadPoolExecutor to process each R value in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_R_value, R, words, top_100_dict) for R in R_values]
        for future in concurrent.futures.as_completed(futures):
            R, intersection_min, intersection_median, intersection_sketch = future.result()
            intersections_min.append((R, intersection_min))
            intersections_median.append((R, intersection_median))
            intersections_sketch.append((R, intersection_sketch))

    # Sort the results by R values
    intersections_min.sort()
    intersections_median.sort()
    intersections_sketch.sort()

    # Extract the sorted intersections
    R_values_sorted = [R for R, _ in intersections_min]
    intersections_min_sorted = [intersection for _, intersection in intersections_min]
    intersections_median_sorted = [intersection for _, intersection in intersections_median]
    intersections_sketch_sorted = [intersection for _, intersection in intersections_sketch]

    # plot the intersections
    plt.figure(figsize=(10, 6))
    plt.plot(R_values_sorted, intersections_min_sorted, label="Count-Min Sketch", color='r')
    plt.plot(R_values_sorted, intersections_median_sorted, label="Count-Median Sketch", color='g')
    plt.plot(R_values_sorted, intersections_sketch_sorted, label="Count-Sketch", color='b')

    plt.xlabel('R')
    plt.ylabel('Intersection')
    plt.title('Intersection of Top 500 Words in Sketch and Top 100 Words in Dictionary')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # plot the errors
    plot_errors(words)

    # plot the intersection of the top 500 words in the sketch and the top 100 words in the dictionary
    plot_top_500_intersection(words)


if __name__ == "__main__":
    main()