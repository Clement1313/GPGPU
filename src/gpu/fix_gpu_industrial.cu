#include "fix_gpu_industrial.cuh"

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/distance.h>
#include <thrust/find.h>
#include <cub/cub.cuh>

struct is_not_garbage {
    __device__ bool operator()(int x) const {
        return x != -27;
    }
};

struct map_pixel {
    __device__ int operator()(int pixel, int index) const {
        if (index % 4 == 0)
            pixel += 1;
        else if (index % 4 == 1)
            pixel -= 5;
        else if (index % 4 == 2)
            pixel += 3;
        else if (index % 4 == 3)
            pixel -= 8;
        
            return pixel;
    } 
};

struct not_zero {
    __device__ bool operator() (int pixel) const {
        return pixel != 0;   
    }
};

struct equalize_pixel {
    const int* histo;
    int cdf_min;
    int image_size;

    __device__ int operator() (int pixel) const {
        return roundf((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min) * 255.f);
    }
};

void fix_image_industrial(Image& to_fix) {

    // #1 Compact
    const int image_size = to_fix.width * to_fix.height;
    thrust::device_vector<int> device_to_fix(to_fix.buffer, to_fix.buffer + to_fix.size());
    thrust::device_vector<int> device_fixed(to_fix.size());

    auto end_it = thrust::copy_if(
        device_to_fix.begin(), device_to_fix.end(),
        device_fixed.begin(),
        is_not_garbage()
    );

    int compact_size = end_it - device_fixed.begin();
    device_fixed.resize(compact_size);

    assert(compact_size == image_size);
    

    // #2 Apply map to fix pixels
    thrust::transform(device_fixed.begin(), device_fixed.end(), thrust::counting_iterator<int>(0), device_fixed.begin(), map_pixel());
    
    
    // #3 Histogram equalization
    
    // Histogram
    const int* in_ptr = thrust::raw_pointer_cast(device_fixed.data());
    int num_items = device_fixed.size();
    thrust::device_vector<int> histogram(256);
    int* out_ptr = thrust::raw_pointer_cast(histogram.data());

    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceHistogram::HistogramEven(temp_storage, temp_storage_bytes, in_ptr, out_ptr, 257, 0, 256, num_items);
    thrust::device_vector<char> temp_buffer(temp_storage_bytes);
    temp_storage = thrust::raw_pointer_cast(temp_buffer.data());

    cub::DeviceHistogram::HistogramEven(temp_storage, temp_storage_bytes, in_ptr, out_ptr, 257, 0, 256, num_items);

    
    // Compute the inclusive sum scan of the histogram
    temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes, out_ptr, out_ptr, 256);

    thrust::device_vector<char> temp_buffer2(temp_storage_bytes);
    temp_storage = thrust::raw_pointer_cast(temp_buffer2.data());
    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes, out_ptr, out_ptr, 256);
    
    
    // Find the first non-zero value in the cumulative histogram
    auto first_none_zero = thrust::find_if(histogram.begin(), histogram.end(), not_zero());
    
    const int cdf_min = *first_none_zero;

    // Apply the map transformation of the histogram equalization
    thrust::transform(device_fixed.begin(), device_fixed.end(), device_fixed.begin(), equalize_pixel{out_ptr, cdf_min, image_size});

    thrust::copy(device_fixed.begin(), device_fixed.end(), to_fix.buffer);

    // Compute total
    uint64_t sum = thrust::reduce(device_fixed.begin(), device_fixed.end(), uint64_t(0));
    to_fix.to_sort.total = sum;
}

// int  compute_total_industrial(const Image& img);