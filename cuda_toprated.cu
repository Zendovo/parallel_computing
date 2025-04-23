#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "rapidjson/document.h"

using namespace rapidjson;

// CUDA kernel for aggregation
__global__ void aggregate(int *product_ids, float *ratings, float *sums, int *counts, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int pid = product_ids[idx];
        float rating = ratings[idx];
        atomicAdd(&sums[pid], rating);
        atomicAdd(&counts[pid], 1);
    }
}

// CUDA kernel for average computation
__global__ void compute_average(float *sums, int *counts, float *averages, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M && counts[idx] > 0)
    {
        averages[idx] = sums[idx] / counts[idx];
    }
}

int main()
{
    std::unordered_map<std::string, int> asin_to_id;
    std::vector<std::string> id_to_asin;
    std::vector<int> product_ids;
    std::vector<float> ratings;

    // Parse and Map ASIN to product_id
    std::ifstream infile("reviews.csv");
    std::string line;
    std::getline(infile, line);

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

    // Allocate GPU memory
    int *d_product_ids, *d_counts;
    float *d_ratings, *d_sums, *d_averages;

    cudaMalloc(&d_product_ids, num_reviews * sizeof(int));
    cudaMalloc(&d_ratings, num_reviews * sizeof(float));
    cudaMalloc(&d_sums, num_products * sizeof(float));
    cudaMalloc(&d_counts, num_products * sizeof(int));
    cudaMalloc(&d_averages, num_products * sizeof(float));

    cudaMemset(d_sums, 0, num_products * sizeof(float));
    cudaMemset(d_counts, 0, num_products * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_product_ids, product_ids.data(), num_reviews * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ratings, ratings.data(), num_reviews * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch Aggregation Kerne
    int blockSize = 256;
    int gridSize = (num_reviews + blockSize - 1) / blockSize;
    aggregate<<<gridSize, blockSize>>>(d_product_ids, d_ratings, d_sums, d_counts, num_reviews);
    cudaDeviceSynchronize();

    // Launch Averaging Kernel
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

    // Copy averages back to CPU
    std::vector<float> averages(num_products);
    cudaMemcpy(averages.data(), d_averages, num_products * sizeof(float), cudaMemcpyDeviceToHost);

    // Find Top-10 Products
    std::vector<std::pair<std::string, float>> result;
    for (int i = 0; i < num_products; ++i)
    {
        if (averages[i] > 0)
        {
            result.emplace_back(id_to_asin[i], averages[i]);
        }
    }

    // Using partial_sort to find top 10 products
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

    // Free GPU memory
    cudaFree(d_product_ids);
    cudaFree(d_ratings);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_averages);

    return 0;
}
