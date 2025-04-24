## Top 10 Rated ASINs

This computational task maps naturally to a **parallel histogram** problem as referenced from R
where each product ID corresponds to a histogram bin. The main challenges are load balancing
and handling concurrent updates to the same bin (i.e., product ID).

### Input Format and Preprocessing:

The original dataset is in JSON format with fields like reviewerID, asin, reviewText, and
overall. For faster parsing and reduced overhead, the dataset is preprocessed to CSV format
containing only two columns:
- asin (Amazon Standard Identification Number, a unique product ID)
- overall (the rating score as a float)
This CSV format reduces parsing complexity and speeds up the loading process for
experimentation and testing.

### Optimized CUDA Kernel Design:

1. **Privatization with Shared Memory Bins** :
    - Each thread block uses shared memory to maintain a local hash table
       (bin_pids, bin_sums, bin_counts) that maps hashed product IDs to
       aggregate values.
    - These shared bins reduce global memory contention by localizing frequent
       writes.
2. **Linear Probing in Shared Memory** :
    - A simple hash function (pid % MAX_BINS_PER_BLOCK) is used.
    - Collisions are resolved with linear probing.
3. **Thread Coarsening with Interleaving** :
    - Each thread is responsible for processing multiple reviews with a difference of
       total_threads, improving memory throughput and latency hiding.
4. **Final Aggregation to Global Memory** :
    - Once all threads finish populating shared bins, the block-level aggregates are
       flushed to global memory using atomicAdd.

### Practical Performance Comparison:

The optimized kernels ran **0.16ms faster than the baseline i.e. 8.6% faster**. The reason for
lack of performance increase could be due to the skew in data being minimal causing very low
contention among the threads.

### Lexicon Sentiment Analyzer

This computational task also maps naturally to a **parallel histogram** problem as referenced
from R6 where each reviewID corresponds to a histogram bin. The main challenges are load
balancing and handling concurrent updates to the same bin (i.e., review ID).

### Input Format and Preprocessing:

The input data is in JSON format, where each line contains a review with a reviewText field.
Preprocessing involves:

- Tokenizing each reviewText using space as a delimiter.
- Mapping each token to its sentiment score using the VADER lexicon.
- Associating each valid score with the corresponding review index.
The result is two arrays:
- host_scores: sentiment scores of valid tokens.
- host_indices: corresponding review indices.
These arrays are then transferred to the GPU for aggregation.

### Optimized CUDA Kernel Design:

**1. Score Aggregation Kernel (aggregateScores):**
    - Shared memory bins are used to locally aggregate sentiment scores per review index
       within each block.
    - Hashing is used to assign each review index to a shared memory bin.
    - Linear probing resolves collisions.
    - After local aggregation, results are flushed to global memory using atomicAdd.
**2. Sentiment Classification Kernel (countSentimentsOptimized):**
    - Shared memory counters tally the number of positive, negative, and neutral reviews per
       block.
    - After local aggregation, the results are added to global memory counters.

### Key CUDA Optimizations:

- Shared Memory Binning : Reduces contention on global memory by aggregating values
within blocks.
- Hashing with Collision Resolution : Hash review indices to shared memory bins, using
linear probing for conflicts.
- Thread Interleaving : Each thread processes multiple data points in a strided manner to
improve throughput and memory access efficiency.
- Atomic Operations : Used for correctness during concurrent updates in both shared and
global memory.

### Practical Performance Comparison:

There was a decent increase in the performance of the optimized version ignoring the overhead
(500ms runtime was because of GPU initialization overhead, to measure the performance
without it the kernel was ran multiple times in the same code and best of 2 were averaged), on
an average, **a 0.5ms improvement was recorded i.e. 23% improvement.**


### Elaborate Reviewers List

Elaborate reviewers are defined as those who have written at least 5 reviews with 50 or more
words each. This task requires parsing a large dataset, analyzing the review text, and
maintaining counts per reviewer, making it suitable for parallelization.
The implementation involves two versions:

1. A sequential C implementation
2. An OpenMP-accelerated implementation targeting multi-core CPU parallelism

### Input Format and Processing:

The input dataset contains Amazon product reviews in CSV format with at least two columns:
- reviewerID: A unique identifier for each reviewer
- reviewText: The text content of the review

#### The processing involves:

- Parsing each review line
- Counting words in the review text
- Tracking elaborate reviews per reviewer using a hash table
- Identifying reviewers who meet the criteria (≥5 reviews with ≥50 words)

### Sequential Implementation Design:

The sequential implementation (c_elaborate.c) uses the following approach:
- Hash table with linear probing to store reviewer information
- Single-threaded parsing of the CSV file, line by line
- In-memory aggregation of reviewer statistics
- Final traversal to identify elaborate reviewers

#### Key components:

1. **Hash Table** : Stores reviewer IDs mapped to their elaborate review counts
2. **Word Counting** : Simple state machine to count words in review text
3. **CSV Parsing** : Basic tokenization to extract reviewer ID and review text
4. **Final Aggregation** : Traversal of the hash table to identify qualifying reviewers

### OpenMP Implementation Design:

The OpenMP implementation (c_elaborate_openmp.c) extends the sequential version with
parallel processing:

- Thread-safe hash table with bucket-level locks
- Parallel processing of reviews using multiple threads
- Pre-loading of the entire file into memory for parallel access
- Dynamic scheduling to handle load imbalance

#### Key optimizations:

1. **Thread-Safe Hash Table** : Uses fine-grained locking with omp_lock_t per bucket
2. **Bulk Loading** : Reads all lines into memory first for efficient parallel processing
3. **Thread-Local Processing** : Each thread independently processes its assigned reviews

### Practical Performance Comparison:

Testing results show significant performance improvement with OpenMP parallelization:
**Implementation Result (Elaborate
Reviewers)
Execution Time
(seconds)**
Sequential 121,864 452.
OpenMP 121,864 75.
This represents a **speedup of 6.02x** with the OpenMP implementation, demonstrating effective
parallelization of the task. The identical result counts confirm the correctness of both
implementations.

### Conclusion:

The OpenMP implementation successfully parallelizes the elaborate reviewer identification task,
achieving a 6x performance improvement over the sequential version. Both versions produce
identical results, confirming the correctness of the parallel implementation. The effective use of
fine-grained locking, dynamic scheduling, and thread-local processing enables the OpenMP
version to scale well with available processor cores, making it suitable for analyzing larger
datasets efficiently.
The performance could potentially be further improved by:

1. Implementing a more sophisticated hash function to reduce collisions
2. Using thread-local hash tables to eliminate locking overhead
3. Optimizing the word counting algorithm for better cache locality
4. Implementing a parallel file reading mechanism


