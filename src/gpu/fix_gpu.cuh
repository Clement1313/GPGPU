#pragma once
#include "image.hh"

#include <nvtx3/nvtx3.hpp>
#include <cuda/std/span>
#include <cuda/buffer>
#include <cuda/atomic>

void fix_image_gpu(Image& to_fix, cuda::stream& stream, cuda::device_memory_pool_ref& mr);
int compute_total_gpu(const Image& img, cuda::stream& stream, cuda::device_memory_pool_ref& mr);
