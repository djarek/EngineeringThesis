#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "simulation.h"
#include "mainwindow.h"
#include "thread"

auto& operator<<(std::ofstream& out, const Vector& vec)
{
	out << "("<< vec.s[0] << "," << vec.s[1] << ") ";
	return out;
}

auto load_program(const cl::Context& context, const size_t size)
{
	std::ifstream kernels_file("kernels/kernels.cl");
	std::string kernel_sources {"#define SIZE "};
	kernel_sources.append(std::to_string(size));
	std::copy(std::istreambuf_iterator<char>(kernels_file), std::istreambuf_iterator<char>(), std::back_inserter(kernel_sources));

	return cl::Program(context, kernel_sources);
}

void ui_main()
{
	SDL_Init(SDL_INIT_EVERYTHING);
	MainWindow window{640, 640};
	window.event_loop();
	SDL_Quit();
}

int main()
{
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	
	cl::Platform::get(&platforms);
	platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
	
	cl::Context context{devices};
	cl::CommandQueue cmd_queue{context, devices[0]};
	
	cl_uint dim = 128;

	auto program = load_program(context, dim);
	try {
		program.build(devices);
	} catch(...) {
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
		throw;
	}

	ScalarField p;
	
	std::thread ui_thread{ui_main};
	Simulation simulation{cmd_queue, context, dim, program};
	for (int y = 0; y < 2000; ++y) {
		simulation.update();
	}

	ui_thread.join();
	return 0;
}
