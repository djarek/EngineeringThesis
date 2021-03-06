/**
 * FluidSim - a free and open-source interactive fluid flow simulator
 * Copyright (C) 2015  Damian Jarek <damian.jarek93@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <iostream>
#include <vector>
#include <fstream>
#include "simulation.h"
#include "mainwindow.h"
#include "thread"
#include "atomic"

std::atomic<bool> running {true};

static auto load_program(const cl::Context& context, const size_t size)
{
	std::ifstream kernels_file("kernels/kernels.cl");
	std::string kernel_sources {"#define SIZE "};
	kernel_sources.append(std::to_string(size));
	std::copy(std::istreambuf_iterator<char>(kernels_file), std::istreambuf_iterator<char>(),
		  std::back_inserter(kernel_sources));

	return cl::Program(context, kernel_sources);
}

static void ui_main(Channel_ptr<ScalarField> dye_field_to_ui, Channel_ptr<Event> events_from_ui, cl_uint dim)
{
	SDL_Init(SDL_INIT_EVERYTHING);
	MainWindow window{640, 640, dim, dye_field_to_ui, events_from_ui};
	window.event_loop();
	SDL_Quit();
}

int main()
{
	auto dye_field_to_ui = Channel<ScalarField>::make();
	auto events_from_ui = Channel<Event>::make();
	cl_uint dim = 512 + 2;
	const auto workgroup_size = std::min(dim - 2, 256u);
	std::thread ui_thread{ui_main, dye_field_to_ui, events_from_ui, dim};

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

	Simulation simulation{cmd_queue, context, dim, program, dye_field_to_ui, events_from_ui, workgroup_size};
	while (running.load(std::memory_order_relaxed)) {
		simulation.update();
	}

	ui_thread.join();
	return 0;
}
