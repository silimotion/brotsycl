#include <stdio.h>
#include <stdlib.h>
#include <CL/sycl.hpp>
#include <iostream>
#include <optional>
#include <fstream>

using namespace cl::sycl;

typedef struct Pixel
{
    char r;
    char g;
    char b;
} Pixel;

void invalid_input_print();

int main(int argc, char *argv[])
{
    double x_size, y_size;            // Size of each pixel
    int canvas_x_size, canvas_y_size; // Size of the canvas
    double x_offset, y_offset;        // Left-down corner of the image
    int p;                            // precision (max number of cycles)
    double power = 1;                 // exponent
    bool force_gpu = false;
    bool force_cpu = false;
    char *filename = NULL;
    char *filename_buffer = NULL;
    char *filename_input = NULL;
    Pixel bg = {0, 0, 0};
    cl::sycl::queue q;
    std::optional<buffer<int, 2>> D;

    if (platform::get_platforms().empty() == true)
    {
        std::cout << "No platforms found - Install appropiate drivers";
        exit(-1);
    }
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0)
        {
            invalid_input_print();
            exit(0);
        }
    }
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--gpu") == 0)
        {
            force_gpu = true;
        }
        else if (strcmp(argv[i], "--cpu") == 0)
        {
            force_cpu = true;
        }
        else if (strcmp(argv[i], "--exp") == 0)
        {
            if (argc < (i + 2))
            {
                invalid_input_print();
                exit(-1);
            }
            power = atof(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-o") == 0)
        {
            if (argc < (i + 2))
            {
                invalid_input_print();
                exit(-1);
            }
            filename = argv[i + 1];
        }
        else if (strcmp(argv[i], "-i") == 0)
        {
            if (argc < (i + 2))
            {
                invalid_input_print();
                exit(-1);
            }
            filename_input = argv[i + 1];
        }
        else if (strcmp(argv[i], "--buf") == 0)
        {
            if (argc < (i + 2))
            {
                invalid_input_print();
                exit(-1);
            }
            filename_buffer = argv[i + 1];
        }
        else if (strcmp(argv[i], "--bg") == 0)
        {
            if (argc < (i + 4))
            {
                invalid_input_print();
                exit(-1);
            }

            bg.r = (atoi(argv[i + 1]) % 256);
            bg.g = (atoi(argv[i + 2]) % 256);
            bg.b = (atoi(argv[i + 3]) % 256);
        }
        else if (strcmp(argv[i], "-l") == 0)
        {
            for (auto const &this_platform : platform::get_platforms())
            {
                std::cout << "Found platform: " << this_platform.get_info<info::platform::name>() << std::endl;
                // Loop through available devices in this platform
                for (auto const &this_device : this_platform.get_devices())
                {
                    std::cout << " Device: " << this_device.get_info<info::device::name>() << std::endl;
                }
                std::cout << "\n";
            }
            exit(0);
        }
    }

    if (force_gpu == true)
    {
        q = queue{gpu_selector()};
    }
    else if (force_cpu == true)
    {
        q = queue{cpu_selector()};
    }
    else
    {
        q = queue{default_selector()};
    }
    std::cout << "Running on device: " << q.get_device().get_info<info::device::name>().c_str() << std::endl;

    if (filename == NULL)
    {
        filename = (char *)"brot1.ppm";
    }
    std::ofstream f(filename);

    if (filename_input == NULL)
    {

        if (argc < 7)
        {
            invalid_input_print();
            exit(-1);
        }

        x_offset = atof(argv[1]);
        y_offset = atof(argv[2]);
        canvas_x_size = atof(argv[5]);
        canvas_y_size = atof(argv[4]);
        x_size = atof(argv[3]) / (double)canvas_x_size;
        y_size = x_size;
        p = atoi(argv[6]);

        if (x_size <= 0 || p <= 0 || canvas_x_size <= 0 || canvas_y_size <= 0 || power <= 0 || bg.r < 0 || bg.g < 0 || bg.b < 0)
        {
            invalid_input_print();
            exit(-1);
        }
        D.emplace(range(canvas_x_size, canvas_y_size));

        q.submit([&](handler &h)
                 {
        accessor acc{D.value(), h, write_only, no_init};
        h.parallel_for(range(canvas_x_size, canvas_y_size),[=] (id<2> idx)
        {
            // Mandelbrot series uses z_1 = z_0^2 + c
            double c_real, c_imaginary; //real and imaginary parts of the point we are calculating
            double z_real, z_imaginary; //real and imaginary of the result
            int flag;

            acc[idx] = -1;
            z_real =z_imaginary = flag = 0;

            c_real =        x_offset+x_size*idx[1];
            c_imaginary =   y_offset+y_size*canvas_x_size-y_size*idx[0];

            for (int i = 0; i < p && flag == 0; i++)
            {
            //We need to save the z_real value for computing z_imaginary
                double z_real_buf;
                z_real_buf = z_real;
                z_real = z_real*z_real - z_imaginary*z_imaginary +c_real;
                z_imaginary =  2*z_real_buf*z_imaginary+c_imaginary;
                if ((z_real > 2) || (z_imaginary > 2))
                {
                    acc[idx] = i;
                    flag = 1;
                }
            }
        }); });

        q.wait();
        if (filename_buffer == NULL)
        {
            filename_buffer = (char *)"buffer.buf";
        }
        std::ofstream g(filename_buffer);
        if (!g)
        {
            std::cout << "Error opening file: " << filename_buffer << std::endl;
            exit(-1);
        }

        g.write((char *)&canvas_x_size, sizeof(int));
        g.write((char *)&canvas_y_size, sizeof(int));
        auto buffer = D.value().get_host_access(read_only);
        for (int i = 0; i < canvas_x_size; i++)
        {
            for (int j = 0; j < canvas_y_size; j++)
            {
                g.write((char *)&buffer[i][j], sizeof(int));
            }
        }
        g.close();
    }
    else
    {
        std::ifstream input_file(filename_input, std::ios::binary);
        if (!input_file)
        {
            std::cout << "Error opening file: " << filename_input << std::endl;
            exit(-1);
        }

        input_file.read((char *)&canvas_x_size, sizeof(canvas_x_size));
        input_file.read((char *)&canvas_y_size, sizeof(canvas_y_size));

        if (canvas_x_size <= 0 || canvas_y_size <= 0)
        {
            std::cout << "Invalid file" << std::endl;
            input_file.close();
            exit(-1);
        }
        try
        {
            D.emplace(range(canvas_x_size, canvas_y_size));
        }
        catch (const exception &e)
        {
            std::cout << "Error processing file" << std::endl;
            exit(-1);
        }

        auto input = D.value().get_host_access(write_only, no_init);
        for (int i = 0; i < canvas_x_size; i++)
        {
            for (int j = 0; j < canvas_y_size; j++)
            {
                try
                {
                    input_file.read((char *)&input[i][j], sizeof(int));
                }
                catch (std::ifstream::failure &e)
                {
                    std::cout << "Invalid file" << std::endl;
                    exit(-1);
                }
            }
        }
        input_file.close();
    }
    buffer<Pixel, 2> C(range(canvas_x_size, canvas_y_size));
    q.submit([&](handler &h)
             {
        accessor acc{C, h, write_only, no_init};
        accessor acc2{D.value(), h, read_only};
        h.parallel_for(range(canvas_x_size, canvas_y_size), [=] (id<2> idx) {
            if (acc2[idx]==-1)
            {
                acc[idx].r = bg.r; //Set the initial value for the points, 0 means it has not diverged
                acc[idx].g = bg.g;
                acc[idx].b = bg.b;
            }
            else
            {
                acc[idx].r = (long)hipsycl::sycl::powr((double)acc2[idx], power)%256;
                acc[idx].g = 0;
                acc[idx].b = 0;
            }
            
        }); });
    q.wait();
    f << "P6 " << canvas_y_size << " " << canvas_x_size << " 255" << std::endl;
    auto result = C.get_host_access(read_only);
    for (int i = 0; i < canvas_x_size; i++)
    {
        for (int j = 0; j < canvas_y_size; j++)
        {
            f.write((char *)&result[i][j], sizeof(Pixel));
        }
    }
    f.close();
}

void invalid_input_print()
{
    std::cout << "Usage: brotsycl x_offset y_offset canvas_h_size canvas_v_size cycles [option]..." << std::endl;
    std::cout << "Create a .ppm image of the mandelbrot set at the given coordinates" << std::endl
              << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --cpu               Force the use of cpu instead of other accelerators" << std::endl;
    std::cout << "  --gpu               Force the use of gpu instead of other accelerators" << std::endl;
    std::cout << "  -l                  List all platforms and devices available" << std::endl;
    std::cout << "  -o <filename>       Save the produced image to the indicated filename (by default brot1.ppm)" << std::endl;
    std::cout << "  --exp <exponent>    The exponent used in the powr function to calculate the color for each pixel," << std::endl
              << "                      use values lower than 1 to make slopes in color slower and avoid \"soup\" images." << std::endl;
    std::cout << "  --bg <r> <g> <b>    Sets the color of the pixels that belong to the set (by default pitch black)" << std::endl;
    std::cout << "  --buf <filename>    Sets the name of the file the buffer will be saved in" << std::endl;
    std::cout << "  -i <filename>       Converts the given buffer file to a .ppm image, ignoring image inputs" << std::endl;
    std::cout << "  --help              Show this help text" << std::endl
              << std::endl;
    return;
}