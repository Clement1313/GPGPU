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

enum Flag {
    X_invalid = 0,
    A_aggregate = 1,
    P_prefix = 2,
};

__global__
void kernel_loop_back_scan(cuda::std::span<int> buffer, cuda::std::span<int> flags, cuda::std::span<int> sum, cuda::std::span<int> prefix, int* counter) {
    int tid = threadIdx.x;
    __shared__ int block_id;
    __shared__ int block_offset;
    
    if (tid == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> counter_id(*counter);
        block_id = counter_id.fetch_add(1);
    }

    __syncthreads();

    int id = blockDim.x * block_id + threadIdx.x;

    extern __shared__ int data[];

    if (id < buffer.size())
        data[tid] = buffer[id];
    else
        data[tid] = 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            data[tid] += data[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        sum[block_id] = data[0];
        cuda::atomic_ref<int, cuda::thread_scope_device> flag(flags[block_id]);

        flag.store(Flag::A_aggregate);
        flag.notify_all();

        int agg = 0;
        int loop_back = block_id - 1;
        while (loop_back >= 0)
        {
            cuda::atomic_ref<int, cuda::thread_scope_device> loop_flag(flags[loop_back]);
            int f = loop_flag.load();

            if (f == Flag::X_invalid) {
                loop_flag.wait(Flag::X_invalid);
                f = loop_flag.load();
            }

            if (f == Flag::P_prefix) {
                agg += prefix[loop_back];
                break;
            }
            
            if (f == Flag::A_aggregate) {
                agg += sum[loop_back];
            }

            loop_back--;
        }

        block_offset = agg;
        prefix[block_id] = data[0] + agg;
        flag.store(Flag::P_prefix);
        flag.notify_all();
    }

    __syncthreads();

    if (id < buffer.size())    
        data[tid] = buffer[id];
    else                        
        data[tid] = 0;

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {

        int tmp = data[tid];
        __syncthreads();

        if (tid + s < blockDim.x) {
            data[tid + s] += tmp;
        }
        __syncthreads();
    }

    if (id < buffer.size())
        buffer[id] = data[tid] + block_offset;
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
void kernel_scatter(cuda::std::span<int> buffer, cuda::std::span<int> out, cuda::std::span<int> excl_predicate) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= buffer.size())
        return;

    if (buffer[id] != GARBAGE_VALUE)
        out[excl_predicate[id]] = buffer[id];
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

    int* ptr = buffer.data();
    int pixel = buffer[id];

    buffer[id] = std::roundf(((cdf[pixel] - *min_cdf) / static_cast<float>(image_size - *min_cdf)) * 255.f);
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

    cuda::device_buffer<int> sum(
        stream, 
        mr, 
        nb_block, 
        cuda::no_init
    );

    cuda::device_buffer<int> flags(
        stream, 
        mr, 
        nb_block, 
        cuda::no_init
    );

    cuda::device_buffer<int> prefix(
        stream, 
        mr, 
        nb_block, 
        cuda::no_init
    );

    cudaMemsetAsync(
        flags.data(), 
        Flag::X_invalid, 
        nb_block * sizeof(int), 
        stream.get()
    );

    int counter = 0;

    kernel_loop_back_scan<<<nb_block, block_size, block_size * sizeof(int), buffer.stream().get()>>>(device_predicate, flags, sum, prefix, &counter);

    nvtxRangePop();

    nvtxRangePushA("exclusive scan kernel");

    cuda::device_buffer<int> tmp(
        stream,
        mr,
        size,
        cuda::no_init
    );

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

    cuda::device_buffer<int> fix_buffer(
        stream,
        mr,
        size,
        cuda::no_init
    );

    kernel_scatter<<<nb_block, block_size, 0, stream.get()>>>(buffer, fix_buffer,device_predicate);

    nvtxRangePop();
    // #2 Apply map to fix pixels
    
    int img_nb_block = cuda::ceil_div(image_size, block_size);

    nvtxRangePushA("fix map kernel");

    kernel_fix_map<<<img_nb_block, block_size, 0, stream.get()>>>(fix_buffer, image_size);

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

    kernel_histogram<<<img_nb_block, block_size, block_size * sizeof(int), stream.get()>>>(fix_buffer, histogram, image_size);

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

    kernel_equalization<<<img_nb_block, block_size, 0, stream.get()>>>(fix_buffer, histogram, cdf_min_device.data(), image_size);

    nvtxRangePop();

    nvtxRangePop();

    cudaMemcpyAsync(
        to_fix.buffer, 
        fix_buffer.data(), 
        size * sizeof(int), 
        cudaMemcpyDeviceToHost, 
        stream.get()
    );

    stream.sync();
}

__global__
void kernel_reduce(cuda::std::span<int> buffer, cuda::std::span<int> odata, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int data[];

    if (i < size)
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

    int result;

    nvtxRangePop();

    nvtxRangePushA("reduce kernel block level");

    kernel_reduce<<<nb_block, block_size, block_size * sizeof(int), stream.get()>>>(input_span, tmp_span, input_span.size());

    nvtxRangePop();
    nvtxRangePushA("reduce kernel grid level");
    
    int count = nb_block;

    while (count > 1) {
        const int next_count = cuda::ceil_div(count, block_size);

        kernel_reduce<<<next_count, block_size, block_size * sizeof(int), stream.get()>>>(tmp_span, input_span, count);

        std::swap(input_span, tmp_span);
        count = next_count;
    }

    nvtxRangePop();

    nvtxRangePushA("copy res device to host");

    cudaMemcpyAsync(
        &result,
        tmp_span.data(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream.get()
    );

    nvtxRangePop();

    stream.sync();

    return result;
}