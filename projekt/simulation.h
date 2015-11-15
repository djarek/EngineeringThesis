#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <CL/opencl.h>

using Scalar = cl_float;
using Vector = cl_float2;
using Point  = cl_int2;
using Offset = Point;
using ScalarField = std::vector<Scalar>;
using VectorField = std::vector<Vector>;

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

public:
	Simulation(cl::CommandQueue cmd_queue, const cl::Context& context, cl_uint cell_count, const cl::Program& program);
	void update();
	void get_pressure_field(ScalarField& pressure_out);
private:
	void enqueueBoundaryKernel(cl::CommandQueue& cmd_queue, cl::Kernel& boundary_kernel) const;
	void enqueueInnerKernel(cl::CommandQueue& cmd_queue, const cl::Kernel& kernel) const;
	void calculate_advection();
	void calculate_diffusion();
	void calculate_divergence_w();
	void zero_fill_vector_field(cl::Buffer& field);
	void zero_fill_scalar_field(cl::Buffer& field);
	void calculate_p();
	void apply_scalar_boundary_conditions(cl::Buffer& buffer);
	void apply_vector_boundary_conditions(cl::Buffer& buffer);
	void calculate_gradient_p();
	void calculate_u();
};
