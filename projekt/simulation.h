#ifndef SIMULATION_H
#define SIMULATION_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "typedefs.h"
#include "channel.h"

class Simulation
{
	cl::CommandQueue cmd_queue;

	//scalar fields
	cl::Buffer p; //pressure field
	cl::Buffer temporary_p;
	cl::Buffer divergence_w;
	cl::Buffer dye;

	//vector fields
	cl::Buffer u; //divergence-free velocity field
	cl::Buffer temporary_w;
	cl::Buffer w; //divergent velocity field
	cl::Buffer gradient_p;

	cl_uint cell_count;
	cl_uint total_cell_count;

	cl::Kernel vector_advection_kernel;
	cl::Kernel scalar_advection_kernel;
	cl::Kernel scalar_jacobi_kernel;
	cl::Kernel vector_jacobi_kernel;
	cl::Kernel divergence_kernel;
	cl::Kernel gradient_kernel;
	cl::Kernel subtract_gradient_p_kernel;
	cl::Kernel vector_boundary_kernel;
	cl::Kernel scalar_boundary_kernel;
	cl::Kernel apply_impulse_kernel;
	cl::Kernel add_dye_kernel;
	cl::Kernel dye_boundary_conditions_kernel;

	Channel_ptr<ScalarField> to_ui;
	Channel_ptr<ScalarField> from_ui;
	
	VectorField zero_vector_buffer;
public:
	Simulation(cl::CommandQueue cmd_queue, const cl::Context& context, cl_uint cell_count, const cl::Program& program, Channel_ptr<ScalarField> to_ui, Channel_ptr<ScalarField> from_ui);
	void update();
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
	void advect_dye();
	void apply_impulse();
	void add_dye();
	void apply_dye_boundary_conditions();
};
#endif //SIMULATION_H
