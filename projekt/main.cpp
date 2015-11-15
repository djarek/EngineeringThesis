#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <CL/opencl.h>

using Scalar = cl_float;
using Vector = cl_float2;
using Point  = cl_int2;
using Offset = Point;

std::ofstream& operator << (std::ofstream& out, const Vector& vec)
{
	out << "("<< vec.s[0] << "," << vec.s[1] << ") ";
	return out;
}

class Simulation
{
	cl::CommandQueue cmd_queue;

	//scalar fields
	cl::Buffer p; //pressure field
	cl::Buffer temporary_p;
	cl::Buffer divergence_w;

	//vector fields
	cl::Buffer u; //divergence-free velocity field
	cl::Buffer temporary_w;
	cl::Buffer w; //divergent velocity field
	cl::Buffer gradient_p;

	cl_uint cell_count;

	cl::Kernel advection_kernel;
	cl::Kernel scalar_jacobi_kernel;
	cl::Kernel vector_jacobi_kernel;
	cl::Kernel divergence_kernel;
	cl::Kernel gradient_kernel;
	cl::Kernel subtract_gradient_p_kernel;
	cl::Kernel vector_boundary_kernel;
	cl::Kernel scalar_boundary_kernel;

	std::vector<Scalar> scalar_buffer;
	std::vector<Vector> vector_buffer;
	
public:
	Simulation(cl::CommandQueue cmd_queue, const cl::Context& context, cl_uint cell_count, const cl::Program& program):
		cmd_queue(cmd_queue),
		cell_count(cell_count),
		advection_kernel(program, "advect"),
		scalar_jacobi_kernel(program, "scalar_jacobi_iteration"),
		vector_jacobi_kernel(program, "vector_jacobi_iteration"),
		divergence_kernel(program, "divergence"),
		gradient_kernel(program, "gradient"),
		subtract_gradient_p_kernel(program, "subtract_gradient_p"),
		vector_boundary_kernel(program, "vector_boundary_condition"),
		scalar_boundary_kernel(program, "scalar_boundary_condition"),
		scalar_buffer(cell_count * cell_count, Scalar{0.0f}),
		vector_buffer(cell_count * cell_count, Vector{0.0f, 0.0f})
	{
		u = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};
		w = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};
		gradient_p = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};
		temporary_w = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};

		p = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
		temporary_p = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
		divergence_w = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};

		const cl_float time_step = 0.1f;
		const cl_float dx = 0.01f;
		const cl_float dx_reciprocal = 1/dx;
		const cl_float halved_dx_reciprocal = dx_reciprocal*0.5f;
		const auto dissipation = Vector{0.99, 0.99};
		const cl_float ni = 1.13e-6;

		advection_kernel.setArg(0, u);
		advection_kernel.setArg(1, u);
		advection_kernel.setArg(2, w);
		advection_kernel.setArg(3, dx_reciprocal);
		advection_kernel.setArg(4, time_step);
		advection_kernel.setArg(5, dissipation);
		
		divergence_kernel.setArg(0, w);
		divergence_kernel.setArg(1, divergence_w);
		divergence_kernel.setArg(2, halved_dx_reciprocal);
		
		scalar_jacobi_kernel.setArg(0, p);
		scalar_jacobi_kernel.setArg(1, divergence_w);
		scalar_jacobi_kernel.setArg(2, temporary_p);
		scalar_jacobi_kernel.setArg(3, cl_float{-dx*dx});
		scalar_jacobi_kernel.setArg(4, cl_float{0.25f});

		vector_jacobi_kernel.setArg(0, w);
		vector_jacobi_kernel.setArg(1, u);
		vector_jacobi_kernel.setArg(2, temporary_w);
		const cl_float alpha = dx * dx / (ni * time_step);
		vector_jacobi_kernel.setArg(3, alpha);
		vector_jacobi_kernel.setArg(4, 1/(4 + alpha));

		gradient_kernel.setArg(0, p);
		gradient_kernel.setArg(1, gradient_p);
		gradient_kernel.setArg(2, halved_dx_reciprocal);

		subtract_gradient_p_kernel.setArg(0, w);
		subtract_gradient_p_kernel.setArg(1, gradient_p);
		subtract_gradient_p_kernel.setArg(2, u);
	}

	void enqueueBoundaryKernel(cl::CommandQueue& cmd_queue, cl::Kernel& boundary_kernel) const
	{
		const cl_uint boundary_cell_count = cell_count - 2;
		//Argument 1 is the offset, used to indicate whether we use the cell above/below/to the left/to the right
		//to calculate the boundary value
		//Enqueue the first and last rows boundary calculations
		boundary_kernel.setArg(1, Offset{0, 1});
		cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{1, 0}, cl::NDRange{boundary_cell_count, 1});
		boundary_kernel.setArg(1, Offset{0, -1});
		cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{1, cell_count - 1}, cl::NDRange{boundary_cell_count, 1});

		//Enqueue the first and last columns boundary calculations
		boundary_kernel.setArg(1, Offset{1, 0});
		cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{0, 1}, cl::NDRange{1, boundary_cell_count});
		boundary_kernel.setArg(1, Offset{-1, 0});
		cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{cell_count - 1, 1}, cl::NDRange{1, boundary_cell_count});

		cmd_queue.enqueueBarrierWithWaitList();
	}

	void enqueueInnerKernel(cl::CommandQueue& cmd_queue, const cl::Kernel& kernel) const
	{
		const auto offset = cl::NDRange{1, 1};
		// Subtract 2 from the range in each dimension to account for 2 boundary cells
		// top & bottom or left & right cells
		const auto range  = cl::NDRange{cell_count - 2,  cell_count - 2}; 
		cmd_queue.enqueueNDRangeKernel(kernel, offset, range);
		cmd_queue.enqueueBarrierWithWaitList();
	}

	void calculate_advection()
	{
		advection_kernel.setArg(0, u);
		advection_kernel.setArg(1, u);
		advection_kernel.setArg(2, w);
		enqueueInnerKernel(cmd_queue, advection_kernel);
	}
	
	void calculate_diffusion()
	{
		zero_fill_vector_field(w);
		/*cmd_queue.enqueueCopyBuffer(u, w, 0, 0, sizeof(Vector)*cell_count*cell_count);
		cmd_queue.enqueueBarrierWithWaitList();*/
		vector_jacobi_kernel.setArg(1, u);

		for (int i = 0; i < 100; ++i) {
			apply_vector_boundary_conditions(w);

			vector_jacobi_kernel.setArg(0, w);
			vector_jacobi_kernel.setArg(2, temporary_w);
			enqueueInnerKernel(cmd_queue, vector_jacobi_kernel);

			using std::swap;
			swap(w, temporary_w);
		}

		apply_vector_boundary_conditions(w);
	}

	void calculate_divergence_w()
	{
		divergence_kernel.setArg(0, w);
		enqueueInnerKernel(cmd_queue, divergence_kernel);
	}

	void zero_fill_vector_field(cl::Buffer& field)
	{
		std::fill(vector_buffer.begin(), vector_buffer.end(), Vector{0, 0});
		cl::copy(cmd_queue, vector_buffer.begin(), vector_buffer.end(), field);
		cmd_queue.enqueueBarrierWithWaitList();
	}

	void zero_fill_scalar_field(cl::Buffer& field)
	{
		std::fill(scalar_buffer.begin(), scalar_buffer.end(), Scalar{0});
		cl::copy(cmd_queue, scalar_buffer.begin(), scalar_buffer.end(), field);
		cmd_queue.enqueueBarrierWithWaitList();
	}

	void calculate_p()
	{
		//Zero-out p to improve convergence rate
		zero_fill_scalar_field(p);

		for (int i = 0; i < 100; ++i) {
			apply_scalar_boundary_conditions(p);

			scalar_jacobi_kernel.setArg(0, p);
			scalar_jacobi_kernel.setArg(2, temporary_p);
			enqueueInnerKernel(cmd_queue, scalar_jacobi_kernel);

			using std::swap;
			swap(p, temporary_p);
		}

		apply_scalar_boundary_conditions(p);
	}

	void apply_scalar_boundary_conditions(cl::Buffer& buffer)
	{
		scalar_boundary_kernel.setArg(0, buffer);
		enqueueBoundaryKernel(cmd_queue, scalar_boundary_kernel);
	}

	void apply_vector_boundary_conditions(cl::Buffer& buffer)
	{
		vector_boundary_kernel.setArg(0, buffer);
		enqueueBoundaryKernel(cmd_queue, vector_boundary_kernel);
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
		calculate_diffusion();
		calculate_advection();
		apply_vector_boundary_conditions(w);
		//TODO: Force application
		calculate_divergence_w();
		apply_scalar_boundary_conditions(divergence_w);

		calculate_p();
		calculate_gradient_p();

		apply_vector_boundary_conditions(gradient_p);

		calculate_u();
		apply_vector_boundary_conditions(u);
	}
};

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
	for (int y = 0; y < 2; ++y) {
		sim.update();
	}
	
	return 0;
}
