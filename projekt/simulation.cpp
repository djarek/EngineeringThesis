#include "simulation.h"
#include <algorithm>
#include <iostream>

constexpr auto jacobi_iterations = 200;

Simulation::Simulation(cl::CommandQueue cmd_queue, const cl::Context& context, cl_uint cell_count, const cl::Program& program, Channel_ptr<ScalarField> to_ui, Channel_ptr<ScalarField> from_ui, Channel_ptr<Event> events_from_ui, cl_uint workgroup_size):
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
	from_ui(from_ui),
	events_from_ui(events_from_ui),
	zero_vector_buffer(total_cell_count, Vector{0.0, 0.0}),
	workgroup_size(workgroup_size)
{
	ScalarField scalar_buffer(total_cell_count, Scalar{0.0});

	u = cl::Buffer{context, zero_vector_buffer.begin(), zero_vector_buffer.end(), false};
	dye = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
	w = cl::Buffer{context, zero_vector_buffer.begin(), zero_vector_buffer.end(), false};
	gradient_p = cl::Buffer{context, zero_vector_buffer.begin(), zero_vector_buffer.end(), false};
	temporary_w = cl::Buffer{context, zero_vector_buffer.begin(), zero_vector_buffer.end(), false};

	p = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
	temporary_p = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
	divergence_w = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};

	const Scalar time_step = 1;
	const Scalar dx = 5;
	const Scalar dx_reciprocal = 1 / dx;
	const Scalar halved_dx_reciprocal = dx_reciprocal * 0.5;
	const auto velocity_dissipation = Vector{0.99, 0.99};
	const Scalar dye_dissipation = 1.0;
	const Scalar ni = 1.13e-6;
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
	scalar_advection_kernel.setArg(5, dye_dissipation);

	divergence_kernel.setArg(0, w);
	divergence_kernel.setArg(1, divergence_w);
	divergence_kernel.setArg(2, halved_dx_reciprocal);

	scalar_jacobi_kernel.setArg(0, p);
	scalar_jacobi_kernel.setArg(1, divergence_w);
	scalar_jacobi_kernel.setArg(2, temporary_p);
	scalar_jacobi_kernel.setArg(3, Scalar{-dx * dx});
	scalar_jacobi_kernel.setArg(4, Scalar{0.25});

	vector_jacobi_kernel.setArg(0, w);
	vector_jacobi_kernel.setArg(1, u);
	vector_jacobi_kernel.setArg(2, temporary_w);
	const Scalar alpha = dx * dx / (ni * time_step);
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
	std::srand(42);
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
	const auto range  = cl::NDRange{workgroup_size, workgroup_size};

	for (uint y = 1; y < cell_count - 2; y += workgroup_size) {
		for (uint x = 1; x < cell_count - 2; x += workgroup_size) {
			cmd_queue.enqueueNDRangeKernel(kernel, cl::NDRange{x, y}, range, local_range);
		}
	}

	cmd_queue.enqueueBarrierWithWaitList();
}

void Simulation::calculate_advection()
{
	vector_advection_kernel.setArg(0, w);
	vector_advection_kernel.setArg(1, u);
	vector_advection_kernel.setArg(2, temporary_w);
	enqueueInnerKernel(cmd_queue, vector_advection_kernel);

	std::swap(temporary_w, w);
}

void Simulation::calculate_diffusion()
{
	for (int i = 0; i < jacobi_iterations; ++i) {
		apply_vector_boundary_conditions(w);

		vector_jacobi_kernel.setArg(0, w);
		vector_jacobi_kernel.setArg(1, w);
		vector_jacobi_kernel.setArg(2, temporary_w);
		enqueueInnerKernel(cmd_queue, vector_jacobi_kernel);

		using std::swap;
		swap(w, temporary_w);
	}

	apply_vector_boundary_conditions(w);
}

void Simulation::calculate_divergence_w()
{
	divergence_kernel.setArg(0, w);
	enqueueInnerKernel(cmd_queue, divergence_kernel);
}

void Simulation::zero_fill_vector_field(cl::Buffer& field)
{
	cl::copy(cmd_queue, zero_vector_buffer.begin(), zero_vector_buffer.end(), field);
}

void Simulation::zero_fill_scalar_field(cl::Buffer& field)
{
	cmd_queue.enqueueFillBuffer(field, Scalar{0.0}, 0, total_cell_count);
}

void Simulation::calculate_p()
{
    zero_fill_scalar_field(p);
	scalar_jacobi_kernel.setArg(1, divergence_w);

	for (int i = 0; i < jacobi_iterations; ++i) {
		apply_scalar_boundary_conditions(p);

		scalar_jacobi_kernel.setArg(0, p);
		scalar_jacobi_kernel.setArg(2, temporary_p);
		enqueueInnerKernel(cmd_queue, scalar_jacobi_kernel);

		using std::swap;
		swap(p, temporary_p);
	}

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
	scalar_advection_kernel.setArg(0, dye);
	scalar_advection_kernel.setArg(1, u);
	scalar_advection_kernel.setArg(2, temporary_p);
	enqueueInnerKernel(cmd_queue, scalar_advection_kernel);

	std::swap(dye, temporary_p);
}

void Simulation::apply_impulse(const Event& simulation_event)
{
	//Vector force{50.0 * (1.0 * std::rand() / RAND_MAX - 0.5), 50.0 * (1.0 * std::rand() / RAND_MAX - 0.5)};
	apply_impulse_kernel.setArg(0, w);
	apply_impulse_kernel.setArg(1, simulation_event.point);
	apply_impulse_kernel.setArg(2, simulation_event.value.as_vector);
	
	apply_impulse_kernel.setArg(3, Scalar{2});
	enqueueInnerKernel(cmd_queue, apply_impulse_kernel);
}

void Simulation::add_dye(const Event& simulation_event)
{
	//Point center{static_cast<cl_int>(cell_count/2), static_cast<cl_int>(cell_count/2)};
	add_dye_kernel.setArg(0, dye);
	add_dye_kernel.setArg(1, simulation_event.point);
	add_dye_kernel.setArg(2, simulation_event.value.as_scalar);
	add_dye_kernel.setArg(3, Scalar{16});
	enqueueInnerKernel(cmd_queue, add_dye_kernel);
}

void print_vector(cl::CommandQueue cmd_queue, const cl::Buffer& buffer, uint cell_count)
{
	VectorField vec;
	vec.resize(cell_count * cell_count, Vector{1, 1});
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
	vec.resize(cell_count * cell_count, Scalar{1});
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
}

void Simulation::update()
{
	cmd_queue.finish();
	auto events = events_from_ui->try_pop_all();
	for (auto& simulation_event : events) {
		if (simulation_event.type == Event::Type::ADD_DYE) {
			add_dye(simulation_event);
		} else if (simulation_event.type == Event::Type::APPLY_FORCE) {
			//std::cout << "Apply impulse\n";
			apply_impulse(simulation_event);
		}
	}
	apply_dye_boundary_conditions();

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

	apply_vector_boundary_conditions(gradient_p);

	calculate_u();
	apply_vector_boundary_conditions(u);

	ScalarField output_buffer;

	if (not from_ui->try_pop(output_buffer)) {
		output_buffer.resize(total_cell_count);
	}

	cl::copy(cmd_queue, dye, output_buffer.begin(), output_buffer.end());

	
	if (dye_buffers_wait_list.empty()) {
		to_ui->try_push(output_buffer);
	} else {
		dye_buffers_wait_list.emplace_back(output_buffer);
		to_ui->try_push_all(dye_buffers_wait_list);
	}
	
	
}
