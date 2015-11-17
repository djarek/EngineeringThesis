#include "simulation.h"
#include <thread>
#include <algorithm>

constexpr auto jacobi_iterations = 100;

Simulation::Simulation(cl::CommandQueue cmd_queue, const cl::Context& context, cl_uint cell_count, const cl::Program& program, Channel_ptr<ScalarField> to_ui, Channel_ptr<ScalarField> from_ui):
	cmd_queue(cmd_queue),
	cell_count(cell_count),
	total_cell_count(cell_count * cell_count),
	vector_advection_kernel(program, "advect_vector"),
	scalar_advection_kernel(program, "advect_scalar"),
	scalar_jacobi_kernel(program, "scalar_jacobi_iteration"),
	vector_jacobi_kernel(program, "vector_jacobi_iteration"),
	divergence_kernel(program, "divergence"),
	gradient_kernel(program, "gradient"),
	subtract_gradient_p_kernel(program, "subtract_gradient_p"),
	vector_boundary_kernel(program, "vector_boundary_condition"),
	scalar_boundary_kernel(program, "scalar_boundary_condition"),
	apply_impulse_kernel(program, "apply_impulse"),
	add_dye_kernel(program, "add_dye"),
	dye_boundary_conditions_kernel(program, "apply_dye_boundary_conditions"),
	to_ui(to_ui),
	from_ui(from_ui)
{
	ScalarField scalar_buffer(total_cell_count, Scalar{0.0});
	VectorField vector_buffer(total_cell_count, Vector{0.0, 0.0});

	u = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};
	dye = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
	w = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};
	gradient_p = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};
	temporary_w = cl::Buffer{context, vector_buffer.begin(), vector_buffer.end(), false};

	p = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
	temporary_p = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
	divergence_w = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};

	const cl_float time_step = 1;
	const cl_float dx = 1;
	const cl_float dx_reciprocal = 1 / dx;
	const cl_float halved_dx_reciprocal = dx_reciprocal * 0.5;
	const auto velocity_dissipation = Vector{0.95, 0.95};
	const auto dye_dissipation = Vector{0.999, 0.999};
	const cl_float ni = 1.13e-9;
	vector_advection_kernel.setArg(0, u);
	vector_advection_kernel.setArg(1, u);
	vector_advection_kernel.setArg(2, w);
	vector_advection_kernel.setArg(3, dx_reciprocal);
	vector_advection_kernel.setArg(4, time_step);
	vector_advection_kernel.setArg(5, velocity_dissipation);
	
	scalar_advection_kernel.setArg(0, dye);
	scalar_advection_kernel.setArg(1, u);
	scalar_advection_kernel.setArg(2, temporary_p);
	scalar_advection_kernel.setArg(3, dx_reciprocal);
	scalar_advection_kernel.setArg(4, time_step);
	scalar_advection_kernel.setArg(5, dye_dissipation.s[0]);
	
	divergence_kernel.setArg(0, w);
	divergence_kernel.setArg(1, divergence_w);
	divergence_kernel.setArg(2, halved_dx_reciprocal);
	
	scalar_jacobi_kernel.setArg(0, p);
	scalar_jacobi_kernel.setArg(1, divergence_w);
	scalar_jacobi_kernel.setArg(2, temporary_p);
	scalar_jacobi_kernel.setArg(3, cl_float{-dx * dx});
	scalar_jacobi_kernel.setArg(4, cl_float{0.25});

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

	apply_impulse_kernel.setArg(4, time_step);
	
	add_dye_kernel.setArg(4, time_step);
}

const auto local_range = cl::NullRange;

void Simulation::enqueueBoundaryKernel(cl::CommandQueue& cmd_queue, cl::Kernel& boundary_kernel) const
{
	cmd_queue.finish();
	const cl_uint boundary_cell_count = cell_count - 2;
	//Argument 1 is the offset, used to indicate whether we use the cell above/below/to the left/to the right
	//to calculate the boundary value
	//Enqueue the first and last rows boundary calculations
	boundary_kernel.setArg(1, Offset{0, 1});
	cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{1, 0}, cl::NDRange{boundary_cell_count, 1}, local_range);
	boundary_kernel.setArg(1, Offset{0, -1});
	cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{1, cell_count - 1}, cl::NDRange{boundary_cell_count, 1}, local_range);

	//Enqueue the first and last columns boundary calculations
	boundary_kernel.setArg(1, Offset{1, 0});
	cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{0, 1}, cl::NDRange{1, boundary_cell_count}, local_range);
	boundary_kernel.setArg(1, Offset{-1, 0});
	cmd_queue.enqueueNDRangeKernel(boundary_kernel, cl::NDRange{cell_count - 1, 1}, cl::NDRange{1, boundary_cell_count}, local_range);

	cmd_queue.enqueueBarrierWithWaitList();
}

void Simulation::enqueueInnerKernel(cl::CommandQueue& cmd_queue, const cl::Kernel& kernel) const
{
	// Subtract 2 from the range in each dimension to account for 2 boundary cells
	// top & bottom or left & right cells
	const auto range  = cl::NDRange{32, 32};
	
	for (uint y = 1; y < cell_count - 2; y += 32) {
		for (uint x = 1; x < cell_count - 2; x += 32) {
			cmd_queue.enqueueNDRangeKernel(kernel, cl::NDRange{x, y}, range, local_range);
		}
	}
	
	//cmd_queue.enqueueNDRangeKernel(kernel, offset, range, local_range);
	cmd_queue.enqueueBarrierWithWaitList();
}

void Simulation::calculate_advection()
{
	vector_advection_kernel.setArg(0, w);
	vector_advection_kernel.setArg(1, u);
	vector_advection_kernel.setArg(2, temporary_w);
	enqueueInnerKernel(cmd_queue, vector_advection_kernel);
	cmd_queue.finish();
	std::swap(temporary_w, w);
}

void Simulation::calculate_diffusion()
{
	//zero_fill_vector_field(w);
	vector_jacobi_kernel.setArg(1, w);
	cmd_queue.finish();

	for (int i = 0; i < jacobi_iterations; ++i) {
		apply_vector_boundary_conditions(w);
		cmd_queue.finish();

		vector_jacobi_kernel.setArg(0, w);
		vector_jacobi_kernel.setArg(2, temporary_w);
		enqueueInnerKernel(cmd_queue, vector_jacobi_kernel);

		cmd_queue.finish();
		using std::swap;
		swap(w, temporary_w);
	}

	cmd_queue.finish();
	apply_vector_boundary_conditions(w);
}

void Simulation::calculate_divergence_w()
{
	divergence_kernel.setArg(0, w);
	enqueueInnerKernel(cmd_queue, divergence_kernel);
}

void Simulation::zero_fill_vector_field(cl::Buffer& field)
{
	VectorField vector_buffer(total_cell_count, Vector{0.0, 0.0});
	cl::copy(cmd_queue, vector_buffer.begin(), vector_buffer.end(), field);
	//cmd_queue.enqueueFillBuffer(field, Vector{0.0, 0.0}, 0, total_cell_count);
}

void Simulation::zero_fill_scalar_field(cl::Buffer& field)
{
	cmd_queue.enqueueFillBuffer(field, Scalar{0.0}, 0, total_cell_count);
}

void Simulation::calculate_p()
{
	//Zero-out p to improve convergence rate
	zero_fill_scalar_field(p);
	cmd_queue.finish();

	for (int i = 0; i < jacobi_iterations; ++i) {
		apply_scalar_boundary_conditions(p);
		cmd_queue.finish();

		scalar_jacobi_kernel.setArg(0, p);
		scalar_jacobi_kernel.setArg(1, divergence_w);
		scalar_jacobi_kernel.setArg(2, temporary_p);
		enqueueInnerKernel(cmd_queue, scalar_jacobi_kernel);
		cmd_queue.finish();

		using std::swap;
		swap(p, temporary_p);
	}

	cmd_queue.finish();
	apply_scalar_boundary_conditions(p);
}

void Simulation::apply_scalar_boundary_conditions(cl::Buffer& buffer)
{
	scalar_boundary_kernel.setArg(0, buffer);
	enqueueBoundaryKernel(cmd_queue, scalar_boundary_kernel);
}

void Simulation::apply_vector_boundary_conditions(cl::Buffer& buffer)
{
	vector_boundary_kernel.setArg(0, buffer);
	enqueueBoundaryKernel(cmd_queue, vector_boundary_kernel);
}

void Simulation::calculate_gradient_p()
{
	gradient_kernel.setArg(0, p);
	enqueueInnerKernel(cmd_queue, gradient_kernel);
}

void Simulation::calculate_u()
{
	enqueueInnerKernel(cmd_queue, subtract_gradient_p_kernel);
}

void Simulation::advect_dye()
{
	cmd_queue.finish();
	scalar_advection_kernel.setArg(0, dye);
	scalar_advection_kernel.setArg(1, u);
	scalar_advection_kernel.setArg(2, temporary_p);
	zero_fill_scalar_field(temporary_p);
	enqueueInnerKernel(cmd_queue, scalar_advection_kernel);
	cmd_queue.finish();
	std::swap(dye, temporary_p);
}

void Simulation::apply_impulse()
{
	Point center{cell_count/2, cell_count/2};
	apply_impulse_kernel.setArg(0, w);
	apply_impulse_kernel.setArg(1, center);
	apply_impulse_kernel.setArg(2, Vector{255, 0});
	apply_impulse_kernel.setArg(3, cl_float{4});
	enqueueInnerKernel(cmd_queue, apply_impulse_kernel);
}

void Simulation::add_dye()
{
	//Point center{cell_count/2, cell_count/2};
	Point center{cell_count/2, cell_count/2};
	add_dye_kernel.setArg(0, dye);
	add_dye_kernel.setArg(1, center);
	add_dye_kernel.setArg(2, Scalar{1});
	add_dye_kernel.setArg(3, cl_float{60});
	enqueueInnerKernel(cmd_queue, add_dye_kernel);
}

#include <iostream>

void print_vector(cl::CommandQueue cmd_queue, const cl::Buffer& buffer, uint cell_count)
{
	VectorField vec;
	vec.resize(cell_count*cell_count, Vector{1, 1});
	cmd_queue.finish();
	cl::copy(cmd_queue, buffer, vec.begin(), vec.end());
	cmd_queue.finish();
	for (uint x = 1; x < cell_count - 2; ++x) {
		for (uint y = 1; y < cell_count - 2; ++y) {
			std::cout << "(" << vec[y * cell_count + x].s[0] << ", " << vec[y * cell_count + x].s[1] << ") ";
		}
		std::cout << std::endl;
	}
}

void print_scalar(cl::CommandQueue cmd_queue, const cl::Buffer& buffer, uint cell_count)
{
	ScalarField vec;
	vec.resize(cell_count*cell_count, Scalar{1});
	cmd_queue.finish();
	cl::copy(cmd_queue, buffer, vec.begin(), vec.end());
	cmd_queue.finish();
	for (uint x = 1; x < cell_count - 2; ++x) {
		for (uint y = 1; y < cell_count - 2; ++y) {
			std::cout << vec[y * cell_count + x] << " ";
		}
		std::cout << std::endl;
	}
}

void Simulation::apply_dye_boundary_conditions()
{
	dye_boundary_conditions_kernel.setArg(0, dye);
	enqueueBoundaryKernel(cmd_queue, dye_boundary_conditions_kernel);
	cmd_queue.finish();
	//print_scalar(cmd_queue, dye, cell_count);
}

void Simulation::update()
{
	cmd_queue.finish();

	static bool i = 0;
	if (i % 20 == 0) {
		apply_impulse();
	}

	apply_vector_boundary_conditions(w);

	calculate_advection();
	

	apply_vector_boundary_conditions(w);

	advect_dye();
	
	apply_dye_boundary_conditions();
	calculate_diffusion();
	apply_vector_boundary_conditions(w);

	calculate_divergence_w();
	apply_scalar_boundary_conditions(divergence_w);

	calculate_p();
	calculate_gradient_p();
	//print_vector(cmd_queue, w, cell_count);

	apply_vector_boundary_conditions(gradient_p);

	calculate_u();
	apply_vector_boundary_conditions(u);
	if (i % 20 == 0) {
		add_dye();
		std::cout << "adding dye" << std::endl;
		apply_dye_boundary_conditions();
	}
	++i;
	ScalarField output_buffer;
	
	if (not from_ui->try_pop(output_buffer)) {
		output_buffer.resize(total_cell_count);
	}
	cl::copy(cmd_queue, dye, output_buffer.begin(), output_buffer.end());
	cmd_queue.finish();
	while (not to_ui->try_push(output_buffer));
}