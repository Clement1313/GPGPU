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
            *first_index = buffer[id];
        }
    }
}

__global__
void kernel_equalization(cuda::std::span<int> buffer, cuda::std::span<int> cdf, int* min_cdf, int image_size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= image_size)
        return;

    buffer[id] = std::roundf(((cdf[buffer[id]] - *min_cdf) / static_cast<float>(image_size - *min_cdf)) * 255.f);
}


void fix_image_gpu(Image& to_fix, cuda::stream& stream, cuda::device_memory_pool_ref& mr)
{
    nvtx3::scoped_range range("image fix");

    const int image_size = to_fix.width * to_fix.height;

    // CUDA

    const int size = to_fix.size();

    int block_size = 1024;
    int nb_block = cuda::ceil_div(size, block_size);


    // #1 Compact

    // Build predicate vector

    nvtxRangePushA("init variable");

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

    stream.sync();
    nvtxRangePop();

    nvtxRangePushA("predicate kernel");

    kernel_predicate<<<nb_block, block_size, 0, stream.get()>>>(buffer, device_predicate);

    nvtxRangePop();

    // Compute the exclusive sum of the predicate

    nvtxRangePushA("scan kernel");

    cuda::device_buffer<int> tmp(
        stream,
        mr,
        size,
        cuda::no_init
    );

    kernel_scan<<<nb_block, block_size, 0, stream.get()>>>(device_predicate, tmp);

    kernel_scan_aux<<<cuda::ceil_div(nb_block, block_size), block_size, 0, stream.get()>>>(tmp);

    propagate<<<nb_block, block_size, 0, stream.get()>>>(device_predicate, tmp);

    nvtxRangePop();

    nvtxRangePushA("exclusive scan kernel");

    cudaMemcpyAsync(
        tmp.data(), 
        device_predicate.data(), 
        size * sizeof(int), 
        cudaMemcpyDeviceToDevice, 
        stream.get()
    );

    kernel_exclusive_scan<<<nb_block, block_size, 0, stream.get()>>>(device_predicate, tmp);

    nvtxRangePop();

    // Scatter to the corresponding addresses

    nvtxRangePushA("scatter kernel");

    kernel_scatter<<<nb_block, block_size, 0, stream.get()>>>(buffer, device_predicate);

    nvtxRangePop();
    // #2 Apply map to fix pixels
    
    int img_nb_block = cuda::ceil_div(image_size, block_size);

    nvtxRangePushA("fix map kernel");

    kernel_fix_map<<<img_nb_block, block_size, 0, stream.get()>>>(buffer, image_size);

    nvtxRangePop();

    // #3 Histogram equalization

    // Histogram
    nvtxRangePushA("histogram kernels");

    nvtxRangePushA("histogram kernel");

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

    nvtxRangePop();

    // Compute the inclusive sum scan of the histogram
    // Find the first non-zero value in the cumulative histogram

    nvtxRangePushA("inclusive scan kernel");

    cuda::device_buffer<int> cdf_min_device(
        stream,
        mr,
        1,
        cuda::no_init
    );

    kernel_inclusive_scan<<<1, HISTOGRAM_SIZE, 0, stream.get()>>>(histogram, cdf_min_device.data());  

    nvtxRangePop();

    // Apply the map transformation of the histogram equalization
    nvtxRangePushA("equalization kernel");

    kernel_equalization<<<img_nb_block, block_size, 0, stream.get()>>>(buffer, histogram, cdf_min_device.data(), image_size);

    nvtxRangePop();

    nvtxRangePop();

    cudaMemcpyAsync(
        to_fix.buffer, 
        buffer.data(), 
        size * sizeof(int), 
        cudaMemcpyDeviceToHost, 
        stream.get()
    );

    stream.sync();
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

int compute_total_gpu(const Image& img, cuda::stream& stream, cuda::device_memory_pool_ref& mr) {

    nvtx3::scoped_range range("image reduce");

    int size = img.width * img.height;;

    nvtxRangePushA("variable init");

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

    int nb_block_reduce = cuda::ceil_div(nb_block, block_size);
    int result;

    nvtxRangePop();

    nvtxRangePushA("reduce kernel block level");

    kernel_reduce<<<nb_block, block_size, block_size * sizeof(int), stream.get()>>>(input_span, tmp_span);

    nvtxRangePop();
    nvtxRangePushA("reduce kernel grid level");
    
    kernel_reduce<<<nb_block_reduce, block_size, block_size * sizeof(int), stream.get()>>>(tmp_span, tmp_span);

    nvtxRangePop();

    nvtxRangePushA("copy res device to host");

    cudaMemcpyAsync(
        &result,
        tmp.data(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream.get()
    );

    nvtxRangePop();

    stream.sync();

    return result;
}