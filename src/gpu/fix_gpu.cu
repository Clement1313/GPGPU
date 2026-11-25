#include "fix_cpu.cuh"
#include "image.hh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <nvtx3/nvtx3.hpp>
#include <cuda/std/span>
#include <cuda/buffer>
#include <cuda/atomic>

constexpr int GARBAGE_VALUE = -27;
constexpr int HISTOGRAM_SIZE = 256;

__global__
void kernel_predicate(cuda::std::span<int> buffer, cuda::std::span<int> predicate) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < buffer.size())
        predicate[id] = buffer[id] != GARBAGE_VALUE;
}

__global__
void kernel_scan(cuda::std::span<int> buffer, cuda::std::span<int> out)
{
    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int s = 1; s < blockDim.x; s *= 2) {

        int tmp = id < buffer.size() ? buffer[id] : 0;
        __syncthreads();

        if (tid + s < blockDim.x && id + s < buffer.size()) {
            buffer[id + s] += tmp;
        }
        __syncthreads();
    }

    if (tid == 0) {
        int last = min(blockDim.x * (blockIdx.x + 1), (int)buffer.size()) - 1;
        out[blockIdx.x] = buffer[last];
    } 
}

__global__
void kernel_scan_aux(cuda::std::span<int> buffer)
{
    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int s = 1; s < blockDim.x; s *= 2) {

        int tmp = id < buffer.size() ? buffer[id] : 0;
        __syncthreads();

        if (tid + s < blockDim.x && id + s < buffer.size()) {
            buffer[id + s] += tmp;
        }
        __syncthreads();
    }
}

__global__
void propagate(cuda::std::span<int> buffer, cuda::std::span<int> val)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    int sum = blockIdx.x != 0 ? val[blockIdx.x - 1] : 0;

    if (id < buffer.size())
        buffer[id] += sum;
}

__global__
void kernel_exclusive_scan(cuda::std::span<int> buffer, cuda::std::span<int> tmp) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id + 1 < buffer.size()) {
        buffer[id + 1] = tmp[id];
    }

    if (id == 0)
        buffer[id] = 0;
}

__global__
void kernel_scatter(cuda::std::span<int> buffer, cuda::std::span<int> excl_predicate) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= buffer.size())
        return;

    int val = buffer[id];

    if (val != GARBAGE_VALUE)
        buffer[excl_predicate[id]] = val;
}

__global__
void kernel_fix_map(cuda::std::span<int> buffer, int image_size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= image_size)
        return;

    if      (id % 4 == 0)   buffer[id] += 1;
    else if (id % 4 == 1)   buffer[id] -= 5;
    else if (id % 4 == 2)   buffer[id] += 3;
    else if (id % 4 == 3)   buffer[id] -= 8;
}

__global__
void kernel_histogram(cuda::std::span<int> buffer, cuda::std::span<int> histogram, int size){
    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ int local_histogram[];

    for (int i = tid; i < HISTOGRAM_SIZE; i += blockDim.x) {
        local_histogram[i] = 0;
    }

    __syncthreads();

    if (id < size) {
        cuda::atomic_ref<int, cuda::thread_scope_device>(local_histogram[buffer[id]]).fetch_add(1);
    }

    __syncthreads();

    for (int i = tid; i < HISTOGRAM_SIZE; i += blockDim.x) {
        cuda::atomic_ref<int, cuda::thread_scope_device>(histogram[i]).fetch_add(local_histogram[i]);
    }
}

__global__
void kernel_inclusive_scan(cuda::std::span<int> buffer, int* first_index)
{
    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int s = 1; s < blockDim.x; s *= 2) {

        int tmp = id < buffer.size() ? buffer[id] : 0;
        __syncthreads();

        if (tid + s < blockDim.x && id + s < buffer.size()) {
            buffer[id + s] += tmp;
        }
        __syncthreads();
    }

    if (id < buffer.size() && buffer[id] != 0) {
        if (id == 0 || buffer[id - 1] == 0) {
            *first_index = id;
        }
    }
}


void fix_image_gpu(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;

    // CUDA

    const int size = to_fix.size();

    cuda::stream stream{cuda::device_ref(0)};
    cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(cuda::device_ref(0));

    int block_size = 1024;
    int nb_block = cuda::ceil_div(size, block_size);


    // #1 Compact

    // Build predicate vector

    cuda::device_buffer<int> buffer(
        stream,
        mr,
        size,
        cuda::no_init
    );

    cudaMemcpyAsync(
        buffer.data(),
        to_fix.buffer,
        size * sizeof(int),
        cudaMemcpyHostToDevice,
        stream.get()
    );

    cuda::device_buffer<int> device_predicate(
        stream,
        mr,
        size,
        cuda::no_init
    );

    kernel_predicate<<<nb_block, block_size, 0, stream.get()>>>(buffer, device_predicate);

    /*
    std::vector<int> predicate(to_fix.size(), 0);

    for (int i = 0; i < to_fix.size(); ++i)
        if (to_fix.buffer[i] != GARBAGE_VALUE)
            predicate[i] = 1;
    */

    // Compute the exclusive sum of the predicate

    cuda::device_buffer<int> tmp(
        stream,
        mr,
        size,
        cuda::no_init
    );

    kernel_scan<<<nb_block, block_size, 0, stream.get()>>>(device_predicate, tmp);

    kernel_scan_aux<<<cuda::ceil_div(nb_block, block_size), block_size, 0, stream.get()>>>(tmp);

    propagate<<<nb_block, block_size, 0, stream.get()>>>(device_predicate, tmp);

    cudaMemcpyAsync(
        tmp.data(), 
        device_predicate.data(), 
        size * sizeof(int), 
        cudaMemcpyDeviceToDevice, 
        stream.get()
    );

    kernel_exclusive_scan<<<nb_block, block_size, 0, stream.get()>>>(device_predicate, tmp);

    //std::exclusive_scan(predicate.begin(), predicate.end(), predicate.begin(), 0);

    // Scatter to the corresponding addresses

    kernel_scatter<<<nb_block, block_size, 0, stream.get()>>>(buffer, device_predicate);

    /*
    for (std::size_t i = 0; i < predicate.size(); ++i)
        if (to_fix.buffer[i] != GARBAGE_VALUE)
            to_fix.buffer[predicate[i]] = to_fix.buffer[i];
    */

    // #2 Apply map to fix pixels
    
    int img_nb_block = cuda::ceil_div(image_size, block_size);

    kernel_fix_map<<<img_nb_block, block_size, 0, stream.get()>>>(buffer, image_size);

    // Enlever une fois cuda histo implemente
    cudaMemcpyAsync(
        to_fix.buffer, 
        buffer.data(), 
        size * sizeof(int), 
        cudaMemcpyDeviceToHost, 
        stream.get()
    );

    /*
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
    */

    // #3 Histogram equalization

    // Histogram

    cuda::device_buffer<int> histogram (
        stream,
        mr,
        HISTOGRAM_SIZE,
        cuda::no_init
    );

    cudaMemsetAsync(
        histogram.data(), 
        0, 
        HISTOGRAM_SIZE * sizeof(int), 
        stream.get()
    );

    kernel_histogram<<<img_nb_block, block_size, block_size * sizeof(int), stream.get()>>>(buffer, histogram, image_size);

    std::array<int, 256> histo;

    /*
    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
        ++histo[to_fix.buffer[i]];
    */

    // Compute the inclusive sum scan of the histogram

    cuda::device_buffer<int> cdf_min_device(
        stream,
        mr,
        1,
        cuda::no_init
    );

    kernel_inclusive_scan<<<1, HISTOGRAM_SIZE, 0, stream.get()>>>(histogram, cdf_min_device.data());

    cudaMemcpyAsync(
        histo.data(),
        histogram.data(),
        HISTOGRAM_SIZE * sizeof(int),
        cudaMemcpyDeviceToHost,
        stream.get()
    );

    

    //std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    // Find the first non-zero value in the cumulative histogram

    int cdf_min;

    cudaMemcpyAsync(
        &cdf_min,
        cdf_min_device.data(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream.get()
    );

    stream.sync();
    /*
    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero;
    */

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
    cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(cuda::device_ref(0));

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