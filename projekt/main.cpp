#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "simulation.h"

std::ofstream& operator << (std::ofstream& out, const Vector& vec)
{
	out << "("<< vec.s[0] << "," << vec.s[1] << ") ";
	return out;
}

cl::Program load_program(const cl::Context& context, const size_t size)
{
	std::ifstream kernels_file("kernels/kernels.cl");
	std::string kernel_sources {"#define SIZE "};
	kernel_sources.append(std::to_string(size));
	std::copy(std::istreambuf_iterator<char>(kernels_file), std::istreambuf_iterator<char>(), std::back_inserter(kernel_sources));

	return cl::Program(context, kernel_sources);
}

int main()
{
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	
	cl::Platform::get(&platforms);
	platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
	
	cl::Context context(devices);
	cl::CommandQueue cmd_queue(context, devices[0]);
	
	cl_uint dim = 128;

	auto program = load_program(context, dim);
	try {
		program.build(devices);
	} catch(...) {
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
		throw;
	}

	Simulation sim{cmd_queue, context, dim, program};
	for (int y = 0; y < 2000; ++y) {
		sim.update();
	}
	
	return 0;
}
