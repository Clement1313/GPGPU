#include "fix_cpu.cuh"
#include "image.hh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <nvtx3/nvtx3.hpp>
#include <cuda/std/span>
#include <cuda/buffer>

void fix_image_gpu(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;

    // #1 Compact

    // Build predicate vector

    std::vector<int> predicate(to_fix.size(), 0);

    constexpr int garbage_val = -27;
    for (int i = 0; i < to_fix.size(); ++i)
        if (to_fix.buffer[i] != garbage_val)
            predicate[i] = 1;

    // Compute the exclusive sum of the predicate

    std::exclusive_scan(predicate.begin(), predicate.end(), predicate.begin(), 0);

    // Scatter to the corresponding addresses

    for (std::size_t i = 0; i < predicate.size(); ++i)
        if (to_fix.buffer[i] != garbage_val)
            to_fix.buffer[predicate[i]] = to_fix.buffer[i];


    // #2 Apply map to fix pixels

    for (int i = 0; i < image_size; ++i)
    {
        if (i % 4 == 0)
            to_fix.buffer[i] += 1;
        else if (i % 4 == 1)
            to_fix.buffer[i] -= 5;
        else if (i % 4 == 2)
            to_fix.buffer[i] += 3;
        else if (i % 4 == 3)
            to_fix.buffer[i] -= 8;
    }

    // #3 Histogram equalization

    // Histogram

    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
        ++histo[to_fix.buffer[i]];

    // Compute the inclusive sum scan of the histogram

    std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    // Find the first non-zero value in the cumulative histogram

    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero;

    // Apply the map transformation of the histogram equalization

    std::transform(to_fix.buffer, to_fix.buffer + image_size, to_fix.buffer,
        [image_size, cdf_min, &histo](int pixel)
            {
                return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
            }
    );
}

__global__
void kernel_reduce(cuda::std::span<int> buffer, cuda::std::span<int> odata) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int data[];

    if (i < buffer.size())
        data[tid] = buffer[i];
    else
        data[tid] = 0;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {

        if (tid < s)
            data[tid] += data[tid + s];

        __syncthreads();
    }

    if (tid == 0) odata[blockIdx.x] = data[0];
}

int compute_total_gpu(const Image& img) {

    nvtx3::scoped_range range("My Reduce");

    cuda::stream stream{cuda::device_ref(0)};
    cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(cuda::devices[0]);

    int size = img.width * img.height;;

    cuda::device_buffer<int> input(
        stream,
        mr,
        size,
        cuda::no_init
    );

    cudaMemcpyAsync(
        input.data(), 
        img.buffer, 
        size * sizeof(int), 
        cudaMemcpyHostToDevice,
        stream.get()
    );

    cuda::std::span<int> input_span = cuda::std::span(input.data(), size);

    int block_size = 1024;

    int nb_block = cuda::ceil_div(size, block_size);

    cuda::device_buffer<int> tmp(
        stream, 
        mr, 
        nb_block, 
        cuda::no_init
    );

    cuda::std::span<int> tmp_span = cuda::std::span(tmp.data(), nb_block);

    kernel_reduce<<<nb_block, block_size, block_size * sizeof(int), stream.get()>>>(input_span, tmp_span);
    
    int nb_block_reduce = cuda::ceil_div(nb_block, block_size);
    kernel_reduce<<<nb_block_reduce, block_size, block_size * sizeof(int), stream.get()>>>(tmp_span, tmp_span);

    int result;

    cudaMemcpyAsync(
        &result,
        tmp.data(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream.get()
    );

    stream.sync();

    return result;
}