#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include "gpu/fix_gpu.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>
#include <string>
#include <cstring>
#include <cstdlib>

#include <nvtx3/nvtx3.hpp>

enum class Mode { Cpu, Gpu, Industrial, Check };

static Mode parse_mode(int argc, char* argv[])
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--mode")
        {
            const std::string m = argv[i + 1];
            if (m == "cpu")   return Mode::Cpu;
            if (m == "gpu")   return Mode::Gpu;
            if (m == "industial")   return Mode::Industrial;
            if (m == "check") return Mode::Check;
            std::cerr << "Unknown mode: " << m << " (cpu|gpu|industial|check)" << std::endl;
            std::exit(1);
        }
    return Mode::Cpu;
}

static std::string parse_input(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--input")
        {
            if (i + 1 < argc)
                return argv[i + 1];

            std::cerr << "Missing path after --input\n";
            std::exit(1);
        }
    }

    std::cerr << "Usage: " << argv[0]
              << " --input <file|directory> --mode <cpu|gpu|industial|check>\n";
    std::exit(1);
}


int main(int argc, char* argv[])
{
    const Mode mode = parse_mode(argc, argv);

    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    namespace fs = std::filesystem;

    const fs::path input = parse_input(argc, argv);
    std::vector<std::string> filepaths;

    if (fs::is_regular_file(input))
    {
        // Single file
        filepaths.push_back(input.string());
    }
    else if (fs::is_directory(input))
    {
        // Directory (recursive)
        for (const auto& entry : fs::recursive_directory_iterator(input))
        {
            if (entry.is_regular_file())
                filepaths.push_back(entry.path().string());
        }
    }
    else
    {
        std::cerr << "Invalid input: " << input << std::endl;
        return 1;
    }

    std::sort(filepaths.begin(), filepaths.end());

    if (filepaths.empty())
    {
        std::cerr << "No files found!" << std::endl;
        return 1;
    }

    std::cout << "Found " << filepaths.size() << " files\n";

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    std::vector<Image> refs;
    if (mode == Mode::Check)
        refs.resize(nb_images);

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course
        images[i] = pipeline.get_image(i);

        switch (mode)
        {
            case Mode::Cpu:
                fix_image_cpu(images[i]);
                break;
            case Mode::Gpu:
                // TODO
                fix_image_gpu(images[i]);
                break;
            case Mode::Industrial:
                // TODO
                // fix_image_industial(images[i]);
                break;
            case Mode::Check:
            {
                refs[i] = images[i];
                const std::size_t bytes = images[i].size() * sizeof(*images[i].buffer);
                refs[i].buffer = static_cast<decltype(refs[i].buffer)>(std::malloc(bytes));
                std::memcpy(refs[i].buffer, images[i].buffer, bytes);
                fix_image_cpu(refs[i]);
                // fix_image_gpu(images[i]);
                break;
            }
        }
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        switch (mode)
        {
            case Mode::Cpu:
                image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);
                break;
            case Mode::Gpu:
                image.to_sort.total = compute_total_gpu(image);
                break;
            case Mode::Check:
                 // TODO
                // image.to_sort.total = compute_total_gpu(image);
                break;
            case Mode::Industrial:
                // TODO :
                // image.to_sort.total = compute_total_industial(image);
                break;
        }
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, &images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    if (mode == Mode::Check)
    {
        int nb_errors = 0;
        for (int i = 0; i < nb_images; ++i)
        {
            const int image_size = images[i].width * images[i].height;
            const int ref_total = std::reduce(refs[i].buffer, refs[i].buffer + image_size, 0);

            if (!std::equal(images[i].buffer, images[i].buffer + image_size, refs[i].buffer))
            {
                std::cerr << "[CHECK] Image #" << i << " : pixels differ" << std::endl;
                ++nb_errors;
            }
            if (images[i].to_sort.total != ref_total)
            {
                std::cerr << "[CHECK] Image #" << i << " : total differs (gpu="
                          << images[i].to_sort.total << ", cpu=" << ref_total << ")" << std::endl;
                ++nb_errors;
            }
        }
        std::cout << (nb_errors == 0 ? "[CHECK] OK, GPU == CPU" : "[CHECK] FAILED") << std::endl;

        for (int i = 0; i < nb_images; ++i)
            free(refs[i].buffer);
    }

    for (const auto& id_sorted : to_sort)
    {
        std::cout << "Image #" << images[id_sorted.id].to_sort.id << " total : " << images[id_sorted.id].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[id_sorted.id].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[id_sorted.id].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // TODO : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i)
        free(images[i].buffer);

    return 0;
}
