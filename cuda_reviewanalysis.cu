#include <cuda.h>
#include <fstream>
#include <iostream>
#include "json.hpp"
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

static const int THREADS_PER_BLOCK = 512;

__global__ void aggregateScores(float *device_scores, int *device_indices,
                                float *device_final_scores, int total_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < total_elements) {
        atomicAdd(&device_final_scores[device_indices[thread_id]], device_scores[thread_id]);
    }
}

__global__ void countSentiments(float *device_final_scores, int *device_counts, int total_reviews) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < total_reviews) {
        float score = device_final_scores[thread_id];
        if (score > 0) {
            atomicAdd(&device_counts[0], 1); // Positive
        } else if (score < 0) {
            atomicAdd(&device_counts[1], 1); // Negative
        } else {
            atomicAdd(&device_counts[2], 1); // Neutral
        }
    }
}

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
    std::ifstream lexicon_input("./vader_lexicon.txt");
    if (!lexicon_input.is_open()) {
        std::cerr << "Error: Unable to open lexicon file" << std::endl;
        return 1;
    }

    std::unordered_map<std::string, float> lexicon_map;

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

    std::ifstream reviews_file("./Electronics_5.json");
    if (!reviews_file.is_open()) {
        std::cerr << "Error: Unable to open reviews file" << std::endl;
        return 1;
    }

    std::vector<float> host_scores;
    std::vector<int> host_indices;

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

    float *device_scores, *device_final_scores, *host_final_scores;
    int *device_indices;

    cudaMalloc(&device_scores, host_scores.size() * sizeof(float));
    cudaMalloc(&device_indices, host_indices.size() * sizeof(int));
    cudaMalloc(&device_final_scores, review_count * sizeof(float));

    cudaMemcpy(device_scores, host_scores.data(), host_scores.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_indices, host_indices.data(), host_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(device_final_scores, 0, review_count * sizeof(float));

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

    host_final_scores = (float *)malloc(review_count * sizeof(float));
    cudaMemcpy(host_final_scores, device_final_scores, review_count * sizeof(float), cudaMemcpyDeviceToHost);

    int *device_counts;
    int host_counts[3] = {0, 0, 0}; // [positive, negative, neutral]

    cudaMalloc(&device_counts, 3 * sizeof(int));
    cudaMemset(device_counts, 0, 3 * sizeof(int));

    blocks = (review_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    countSentiments<<<blocks, THREADS_PER_BLOCK>>>(device_final_scores, device_counts, review_count);

    cudaMemcpy(host_counts, device_counts, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Total reviews: " << review_count << std::endl
              << "Positive: " << host_counts[0] << std::endl
              << "Negative: " << host_counts[1] << std::endl
              << "Neutral: " << host_counts[2] << std::endl;

    cudaFree(device_counts);
    free(host_final_scores);
    cudaFree(device_scores);
    cudaFree(device_indices);
    cudaFree(device_final_scores);

    return 0;
}
