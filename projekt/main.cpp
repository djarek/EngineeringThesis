#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "simulation.h"
#include "mainwindow.h"
#include "thread"
#include "atomic"

std::atomic<bool> running {true};
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

void ui_main(Channel_ptr<ScalarField> to_ui, Channel_ptr<Event> events_from_ui, cl_uint dim)
{
	SDL_Init(SDL_INIT_EVERYTHING);
	MainWindow window{1600, 1040, dim, to_ui, events_from_ui};
	window.event_loop();
	SDL_Quit();
}

int main()
{
	auto to_ui = Channel<ScalarField>::make();
	auto events_from_ui = Channel<Event>::make();
	cl_uint dim = 512 + 2;
	std::thread ui_thread{ui_main, to_ui, events_from_ui, dim};

	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;

	cl::Platform::get(&platforms);
	platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);

	cl::Context context{devices};
	cl::CommandQueue cmd_queue{context, devices[0]};

	auto program = load_program(context, dim);
	try {
		program.build(devices);
	} catch(...) {
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
		throw;
	}

	Simulation simulation{cmd_queue, context, dim, program, to_ui, events_from_ui, std::min(dim - 2, 256u)};
	while (running.load(std::memory_order_relaxed)) {
		simulation.update();
	}

	ui_thread.join();
	return 0;
}
