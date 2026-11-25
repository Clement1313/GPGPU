#include "fix_gpu_industrial.cuh"

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/device.h>
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

    int compact_size = thrust::distance(device_fixed.begin(), end_it);
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
    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::inclusive_scan(temp_storage, temp_storage_bytes, out_ptr, out_ptr, 256);

    thrust::device_vector<char> temp_buffer(temp_storage_bytes);
    temp_storage = thrust::raw_pointer_cast(temp_buffer.data());
    cub::DeviceScan::inclusive_scan(temp_storage, temp_storage_bytes, out_ptr, out_ptr, 256);
    
    
    // // Find the first non-zero value in the cumulative histogram

    // auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    // const int cdf_min = *first_none_zero;

    // // Apply the map transformation of the histogram equalization
    
    // std::transform(to_fix.buffer, to_fix.buffer + image_size, to_fix.buffer,
    //     [image_size, cdf_min, &histo](int pixel)
    //         {
    //             return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
    //         }
    // );



    thrus::.copy(device_fixed.begin(), device_fixed.end(), to_fix.buffer);
}

// int  compute_total_industrial(const Image& img);