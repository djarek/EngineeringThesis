#include <algorithm>
#include <array>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <CL/opencl.h>

using Scalar = cl_float;
using Vector = cl_float2;

const auto cell_count = 4;

void enqueueBoundaryKernel(cl::CommandQueue& cmd_queue, cl::Kernel& boundary_kernel)
{
	const auto boundary_cell_count = cell_count - 2;
	//Argument 1 is the offset, used to indicate whether we use the cell above/below/to the left/to the right
	//to calculate the boundary value

	//Enqueue the first and last rows boundary calculations
	boundary_kernel.setArg(1, Vector{0, 1});
	cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{1, 0}, cl::NDRange{boundary_cell_count, 1});
	boundary_kernel.setArg(1, Vector{0, -1});
	cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{1, cell_count - 1}, cl::NDRange{boundary_cell_count, 1});

	//Enqueue the first and last columns boundary calculations
	boundary_kernel.setArg(1, Vector{1, 0});
	cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{0, 1}, cl::NDRange{1, boundary_cell_count});
	boundary_kernel.setArg(1, Vector{-1, 0});
	cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{cell_count - 1, 1}, cl::NDRange{1, boundary_cell_count});

	cmd_queue.enqueueBarrierWithWaitList();
}

void enqueueInnerKernel(cl::CommandQueue& cmd_queue, cl::Kernel& kernel)
{
	const auto offset = cl::NDRange{1, 1};
	// Subtract 2 from the range in each dimension to account for 2 boundary cells
	// top & bottom or left & right cells
	const auto range  = cl::NDRange{cell_count - 2,  cell_count - 2}; 
	cmd_queue.enqueueNDRangeKernel(kernel, offset, range);
	cmd_queue.enqueueBarrierWithWaitList();
}

class Simulation
{
	cl::CommandQueue cmd_queue;
	
	cl::Buffer p; //pressure field
	cl::Buffer temporary_p;
	cl::Buffer divergence_w;
	
	cl::Buffer u; //divergence-free velocity field
	cl::Buffer w; //divergent velocity field
	cl::Buffer gradient_p;

	cl_int cell_count;
	
	cl::Kernel advection_kernel;
	cl::Kernel scalar_jacobi_kernel;
	cl::Kernel divergence_kernel;
	cl::Kernel gradient_kernel;
	cl::Kernel subtract_gradient_p_kernel;
	cl::Kernel vector_boundary_kernel;
	cl::Kernel scalar_boundary_kernel;

	std::vector<Scalar> scalar_buffer;
public:
	Simulation(cl::CommandQueue cmd_queue, cl::Context context, cl_int cell_count, cl::Program program):
		cmd_queue(cmd_queue),
		cell_count(cell_count),
		advection_kernel(program, "advect"),
		scalar_jacobi_kernel(program, "scalar_jacobi_iteration"),
		divergence_kernel(program, "divergence"),
		gradient_kernel(program, "gradient"),
		subtract_gradient_p_kernel(program, "subtract_gradient_p"),
		vector_boundary_kernel(program, "vector_boundary_condition"),
		scalar_boundary_kernel(program, "scalar_boundary_condition"),
		scalar_buffer(cell_count*cell_count, Scalar{0})
	{
		std::vector<Vector> vector_buffer(cell_count*cell_count, Vector{0, 0});

		u = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};
		p = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
		temporary_p = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};

		w = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};
		gradient_p = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};
		divergence_w = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
		
		const cl_float dx = 0.001f;
		const cl_float dx_reciprocal = 1000.f;
		const auto halved_dx_reciprocal = dx_reciprocal/0.5f;
		
		advection_kernel.setArg(3, dx_reciprocal);
		advection_kernel.setArg(4, cl_float{0.001f});
		
		divergence_kernel.setArg(2, halved_dx_reciprocal);
		
		scalar_jacobi_kernel.setArg(1, divergence_w);
		scalar_jacobi_kernel.setArg(3, cl_float{-dx*dx});
		scalar_jacobi_kernel.setArg(4, cl_float{0.25f});

		gradient_kernel.setArg(1, gradient_p);
		gradient_kernel.setArg(2, halved_dx_reciprocal);
		
		subtract_gradient_p_kernel.setArg(0, w);
		subtract_gradient_p_kernel.setArg(1, gradient_p);
		subtract_gradient_p_kernel.setArg(2, u);
	}
	
	void calculate_advection()
	{
		advection_kernel.setArg(0, u);
		advection_kernel.setArg(1, u);
		advection_kernel.setArg(2, w);

		enqueueInnerKernel(cmd_queue, advection_kernel);
	}
	
	void calculate_divergence_w()
	{
		divergence_kernel.setArg(0, w);
		divergence_kernel.setArg(1, divergence_w);
		enqueueInnerKernel(cmd_queue, divergence_kernel);
	}
	
	void calculate_p()
	{
		//Zero-out p to improve convergence rate
		cl::copy(cmd_queue, scalar_buffer.begin(), scalar_buffer.end(), p);
		for (int i = 0; i < 100; ++i) {
			scalar_boundary_kernel.setArg(0, p);
			enqueueBoundaryKernel(cmd_queue, scalar_boundary_kernel);

			scalar_jacobi_kernel.setArg(0, p);
			scalar_jacobi_kernel.setArg(2, temporary_p);
			enqueueInnerKernel(cmd_queue, scalar_jacobi_kernel);

			using std::swap;
			swap(p, temporary_p);
		}
	}
	
	void calculate_gradient_p()
	{
		gradient_kernel.setArg(0, p);
		enqueueInnerKernel(cmd_queue, gradient_kernel);
	}
	
	void calculate_u()
	{
		enqueueInnerKernel(cmd_queue, subtract_gradient_p_kernel);
	}
	
	void update()
	{
		calculate_advection();
		//TODO: Diffusion
		//TODO: Force application
		calculate_divergence_w();
		calculate_p();
		calculate_gradient_p();
		calculate_u();
	}
};

int main()
{
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	
	cl::Platform::get(&platforms);
	platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
	
	cl::Context context(devices);
	cl::CommandQueue cmd_queue(context, devices[0]);
	
	std::ifstream kernels_file("kernels/kernels.cl");
	std::string kernel_sources;
	std::copy(std::istreambuf_iterator<char>(kernels_file), std::istreambuf_iterator<char>(), std::back_inserter(kernel_sources));
	cl::Program program(context, kernel_sources);
	program.build(devices);
	std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
	cl::Kernel vector_boundary(program, "vector_boundary_condition");
	std::vector<Vector> vec_buffer (16, Vector{1, 1});
	std::vector<Scalar> scalar_buffer (16, Scalar{2});
	
	cl::Buffer buffer(context, vec_buffer.begin(), vec_buffer.end(), false);

	vector_boundary.setArg(0, buffer);
	vector_boundary.setArg(1, cl_int2{0, 0});
	
	enqueueBoundaryKernel(cmd_queue, vector_boundary);
	
	cl::copy(cmd_queue, buffer, vec_buffer.begin(), vec_buffer.end());
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			std::cout << vec_buffer[y*4 + x].s[0] << " ";
		}
		std::cout << std::endl;
	}
	
	
	return 0;
}
