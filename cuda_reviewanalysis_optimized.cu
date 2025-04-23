#include <cuda.h>
#include <fstream>
#include <iostream>
#include "json.hpp"
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

static const int THREADS_PER_BLOCK = 256;

#define MAX_BINS_PER_BLOCK 512  // Max bins per block, configurable

// Hash function to map indices to bins
__device__ __forceinline__ int hash(int idx) {
    return idx % MAX_BINS_PER_BLOCK;  // Simple modulo hash
}

// Optimized kernel to aggregate scores using shared memory and atomic operations
__global__ void aggregateScores(float *device_scores, int *device_indices,
                                          float *device_final_scores, int total_elements) {
    __shared__ int bin_indices[MAX_BINS_PER_BLOCK];  // Store the indices in shared memory
    __shared__ float bin_sums[MAX_BINS_PER_BLOCK];   // Store the partial sums in shared memory
    __shared__ int bin_counts[MAX_BINS_PER_BLOCK];   // Store the counts for each bin

    int tid = threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory bins
    for (int i = tid; i < MAX_BINS_PER_BLOCK; i += blockDim.x) {
        bin_indices[i] = -1;  // Indicating empty slot
        bin_sums[i] = 0.0f;
        bin_counts[i] = 0;
    }
    __syncthreads();

    // Populate shared memory with local bin data
    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
        int bin_idx = device_indices[idx];
        float score = device_scores[idx];
        int h = hash(bin_idx);

        // Linear probing for collision resolution
        while (true) {
            int old = atomicCAS(&bin_indices[h], -1, bin_idx);  // Try to insert the index
            if (old == -1 || old == bin_idx) {
                atomicAdd(&bin_sums[h], score);        // Add score for the bin
                atomicAdd(&bin_counts[h], 1);          // Increment the count for the bin
                break;
            }
            h = (h + 1) % MAX_BINS_PER_BLOCK;  // Handle hash collision (linear probing)
        }
    }
    __syncthreads();

    // Flush the results in shared memory to global memory
    for (int i = tid; i < MAX_BINS_PER_BLOCK; i += blockDim.x) {
        int idx = bin_indices[i];
        if (idx != -1) {
            atomicAdd(&device_final_scores[idx], bin_sums[i]);  // Add partial sums to the global final scores
        }
    }
}

// Optimized kernel to count sentiments (positive, negative, neutral) using shared memory
__global__ void countSentimentsOptimized(float *device_final_scores, int *device_counts, int total_reviews) {
    __shared__ int shared_counts[3]; // [positive, negative, neutral]
    if (threadIdx.x < 3) {
        shared_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = thread_id; idx < total_reviews; idx += stride) {
        float score = device_final_scores[idx];
        if (score > 0) {
            atomicAdd(&shared_counts[0], 1); // Positive
        } else if (score < 0) {
            atomicAdd(&shared_counts[1], 1); // Negative
        } else {
            atomicAdd(&shared_counts[2], 1); // Neutral
        }
    }
    __syncthreads();

    if (threadIdx.x < 3) {
        atomicAdd(&device_counts[threadIdx.x], shared_counts[threadIdx.x]);
    }
}

// Tokenize a string based on a separator
std::vector<std::string> tokenize(const std::string &text, const std::string &separator) {
    std::vector<std::string> result;
    size_t start_pos = 0;
    size_t end_pos = text.find(separator);
    while (end_pos != std::string::npos) {
        result.push_back(text.substr(start_pos, end_pos - start_pos));
        start_pos = end_pos + separator.length();
        end_pos = text.find(separator, start_pos);
    }
    result.push_back(text.substr(start_pos));
    return result;
}

int main() {
    // Load sentiment lexicon from file into a hash map
    std::ifstream lexicon_input("./vader_lexicon.txt");
    if (!lexicon_input.is_open()) {
        std::cerr << "Error: Unable to open lexicon file" << std::endl;
        return 1;
    }

    std::unordered_map<std::string, float> lexicon_map;

    // Parse lexicon file and populate the hash map
    std::string lexicon_line;
    while (std::getline(lexicon_input, lexicon_line)) {
        std::string word, score;
        std::stringstream line_stream(lexicon_line);
        std::getline(line_stream, word, '\t');
        std::getline(line_stream, score, '\t');
        try {
            lexicon_map[word] = std::stof(score);
        } catch (const std::invalid_argument &) {
            std::cerr << "Error: Invalid score for word '" << word << "' with value '" << score << "'" << std::endl;
        }
    }
    lexicon_input.close();

    std::cout << "Lexicon entries loaded: " << lexicon_map.size() << std::endl;

    // Read and process reviews from a JSON file
    std::ifstream reviews_file("./Electronics_5.json");
    if (!reviews_file.is_open()) {
        std::cerr << "Error: Unable to open reviews file" << std::endl;
        return 1;
    }

    std::vector<float> host_scores;
    std::vector<int> host_indices;

    // Parse reviews and calculate sentiment scores
    std::string review_line;
    int review_count = 0;
    const std::string separator = " ";

    while (std::getline(reviews_file, review_line)) {
        try {
            json review_json = json::parse(review_line);

            if (review_json.contains("reviewText")) {
                std::string review_text = review_json["reviewText"];
                std::vector<std::string> words = tokenize(review_text, separator);
                bool has_valid_word = false;

                for (const auto &word : words) {
                    if (lexicon_map.find(word) != lexicon_map.end()) {
                        has_valid_word = true;
                        host_scores.push_back(lexicon_map[word]);
                        host_indices.push_back(review_count);
                    }
                }

                if (has_valid_word) {
                    ++review_count;
                }
            }
        } catch (const json::parse_error &e) {
            std::cerr << "Error: Failed to parse JSON - " << e.what() << std::endl;
        }
    }
    reviews_file.close();

    // Allocate memory on the GPU for scores and indices
    float *device_scores, *device_final_scores, *host_final_scores;
    int *device_indices;

    cudaMalloc(&device_scores, host_scores.size() * sizeof(float));
    cudaMalloc(&device_indices, host_indices.size() * sizeof(int));
    cudaMalloc(&device_final_scores, review_count * sizeof(float));

    cudaMemcpy(device_scores, host_scores.data(), host_scores.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_indices, host_indices.data(), host_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(device_final_scores, 0, review_count * sizeof(float));

    // Launch kernel to aggregate scores
    int blocks = (host_scores.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    aggregateScores<<<blocks, THREADS_PER_BLOCK>>>(device_scores, device_indices, device_final_scores, host_scores.size());

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU aggregation kernel time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results back to host memory
    host_final_scores = (float *)malloc(review_count * sizeof(float));
    cudaMemcpy(host_final_scores, device_final_scores, review_count * sizeof(float), cudaMemcpyDeviceToHost);

    // Allocate memory for sentiment counts and launch kernel to count sentiments
    int *device_counts;
    int host_counts[3] = {0, 0, 0}; // [positive, negative, neutral]

    cudaMalloc(&device_counts, 3 * sizeof(int));
    cudaMemset(device_counts, 0, 3 * sizeof(int));

    blocks = (review_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    countSentimentsOptimized<<<blocks, THREADS_PER_BLOCK>>>(device_final_scores, device_counts, review_count);

    cudaMemcpy(host_counts, device_counts, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // Print sentiment analysis results
    std::cout << "Total reviews: " << review_count << std::endl
              << "Positive: " << host_counts[0] << std::endl
              << "Negative: " << host_counts[1] << std::endl
              << "Neutral: " << host_counts[2] << std::endl;

    // Free GPU and host memory
    cudaFree(device_counts);
    free(host_final_scores);
    cudaFree(device_scores);
    cudaFree(device_indices);
    cudaFree(device_final_scores);

    return 0;
}
