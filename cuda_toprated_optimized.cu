#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "rapidjson/document.h"

#define MAX_BINS_PER_BLOCK 512

using namespace rapidjson;

__device__ __forceinline__ int hash(int pid) {
    return pid % MAX_BINS_PER_BLOCK;
}

// CUDA kernel for aggregation
__global__ void aggregate(int *product_ids, float *ratings, float *sums, int *counts, int N)
{
    __shared__ int bin_pids[MAX_BINS_PER_BLOCK];
    __shared__ float bin_sums[MAX_BINS_PER_BLOCK];
    __shared__ int bin_counts[MAX_BINS_PER_BLOCK];

    int tid = threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory bins
    for (int i = tid; i < MAX_BINS_PER_BLOCK; i += blockDim.x) {
        bin_pids[i] = -1;
        bin_sums[i] = 0.0f;
        bin_counts[i] = 0;
    }
    __syncthreads();

    // Populate local hash map
    for (int idx = thread_id; idx < N; idx += total_threads) {
        int pid = product_ids[idx];
        float rating = ratings[idx];
        int h = hash(pid);

        // Linear probing
        while (true) {
            int old = atomicCAS(&bin_pids[h], -1, pid);
            if (old == -1 || old == pid) {
                atomicAdd(&bin_sums[h], rating);
                atomicAdd(&bin_counts[h], 1);
                break;
            }
            h = (h + 1) % MAX_BINS_PER_BLOCK;
        }
    }
    __syncthreads();

    // Flush per-block shared bins to global memory
    for (int i = tid; i < MAX_BINS_PER_BLOCK; i += blockDim.x) {
        int pid = bin_pids[i];
        if (pid != -1) {
            atomicAdd(&sums[pid], bin_sums[i]);
            atomicAdd(&counts[pid], bin_counts[i]);
        }
    }
}

// CUDA kernel for average computation with thread coarsening
__global__ void compute_average(float *sums, int *counts, float *averages, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // Interleaving stride

    for (int i = idx; i < M; i += stride) {
        if (counts[i] > 0) {
            averages[i] = sums[i] / counts[i];
        }
    }
}

int main()
{
    std::unordered_map<std::string, int> asin_to_id;
    std::vector<std::string> id_to_asin;
    std::vector<int> product_ids;
    std::vector<float> ratings;

    // --- STEP 1: Parse and Map ASIN to product_id ---
    std::ifstream infile("reviews.csv");
    std::string line;
    std::getline(infile, line); // Skip header

    while (std::getline(infile, line))
    {
        size_t comma = line.find(',');
        if (comma == std::string::npos)
            continue;

        std::string asin = line.substr(0, comma);
        float rating = std::stof(line.substr(comma + 1));

        int pid;
        if (asin_to_id.find(asin) == asin_to_id.end())
        {
            pid = asin_to_id.size();
            asin_to_id[asin] = pid;
            id_to_asin.push_back(asin);
        }
        else
        {
            pid = asin_to_id[asin];
        }

        product_ids.push_back(pid);
        ratings.push_back(rating);
    }

    int num_reviews = product_ids.size();
    int num_products = id_to_asin.size();
    std::cout << "Parsed " << num_reviews << " reviews for " << num_products << " products.\n";

    // --- STEP 2: Allocate GPU memory ---
    int *d_product_ids, *d_counts;
    float *d_ratings, *d_sums, *d_averages;

    cudaMalloc(&d_product_ids, num_reviews * sizeof(int));
    cudaMalloc(&d_ratings, num_reviews * sizeof(float));
    cudaMalloc(&d_sums, num_products * sizeof(float));
    cudaMalloc(&d_counts, num_products * sizeof(int));
    cudaMalloc(&d_averages, num_products * sizeof(float));

    cudaMemset(d_sums, 0, num_products * sizeof(float));
    cudaMemset(d_counts, 0, num_products * sizeof(int));

    // --- STEP 3: Copy data to GPU ---
    cudaMemcpy(d_product_ids, product_ids.data(), num_reviews * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ratings, ratings.data(), num_reviews * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // --- STEP 4: Launch Aggregation Kernel ---
    int blockSize = 256;
    int gridSize = (num_reviews + blockSize - 1) / blockSize;
    aggregate<<<gridSize, blockSize>>>(d_product_ids, d_ratings, d_sums, d_counts, num_reviews);
    cudaDeviceSynchronize();

    // --- STEP 5: Launch Averaging Kernel ---
    gridSize = (num_products + blockSize - 1) / blockSize;
    compute_average<<<gridSize, blockSize>>>(d_sums, d_counts, d_averages, num_products);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU aggregation kernel time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // --- STEP 6: Copy averages back to CPU ---
    std::vector<float> averages(num_products);
    cudaMemcpy(averages.data(), d_averages, num_products * sizeof(float), cudaMemcpyDeviceToHost);

    // --- STEP 7: Find Top-10 Products ---
    std::vector<std::pair<std::string, float>> result;
    for (int i = 0; i < num_products; ++i)
    {
        if (averages[i] > 0)
        {
            result.emplace_back(id_to_asin[i], averages[i]);
        }
    }

    std::partial_sort(result.begin(), result.begin() + 10, result.end(),
                      [](const auto &a, const auto &b)
                      {
                          return a.second > b.second;
                      });

    std::cout << "\nTop 10 Rated Products:\n";
    for (int i = 0; i < 10 && i < result.size(); ++i)
    {
        std::cout << result[i].first << " -> " << result[i].second << "\n";
    }

    // --- Cleanup ---
    cudaFree(d_product_ids);
    cudaFree(d_ratings);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_averages);

    return 0;
}