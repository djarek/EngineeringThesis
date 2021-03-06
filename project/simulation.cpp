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

#include "simulation.h"
#include <algorithm>
#include <iostream>

constexpr auto jacobi_iterations = 100;

Simulation::Simulation(cl::CommandQueue cmd_queue,
		       const cl::Context& context,
		       cl_uint cell_count,
		       const cl::Program& program,
		       Channel_ptr<ScalarField> to_ui,
		       Channel_ptr<Event> events_from_ui,
		       cl_uint workgroup_size):
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
	vorticity_kernel(program, "vorticity"),
	apply_vorticity_kernel(program, "apply_voritcity_force"),
	apply_gravity_kernel(program, "apply_gravity"),
	to_ui(to_ui),
	events_from_ui(events_from_ui),
	zero_vector_buffer(total_cell_count, Vector{0.0, 0.0}),
	workgroup_size(workgroup_size)
{
	ScalarField scalar_buffer(total_cell_count, Scalar{0.0});

	u = cl::Buffer{context, zero_vector_buffer.begin(), zero_vector_buffer.end(), false};
	w = cl::Buffer{context, zero_vector_buffer.begin(), zero_vector_buffer.end(), false};
	gradient_p = cl::Buffer{context, zero_vector_buffer.begin(), zero_vector_buffer.end(), false};
	temporary_w = cl::Buffer{context, zero_vector_buffer.begin(), zero_vector_buffer.end(), false};

	p = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
	temporary_p = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
	divergence_w = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};
	dye = cl::Buffer{context, scalar_buffer.begin(), scalar_buffer.end(), false};

	const Scalar time_step = .1;
	const Scalar dx = .2;
	const Scalar dx_reciprocal = 1 / dx;
	const Scalar halved_dx_reciprocal = dx_reciprocal * 0.5;
	const auto velocity_dissipation = Vector{0.99, 0.99};
	const Scalar dye_dissipation = 0.999;
	const Scalar ni = 1.13e-3;
	const Scalar vorticity_confinemnet_scale{0.35};
	const Vector vorticity_dx_scale{vorticity_confinemnet_scale * dx, vorticity_confinemnet_scale * dx};
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

	vorticity_kernel.setArg(0, w);
	vorticity_kernel.setArg(1, temporary_p);
	vorticity_kernel.setArg(2, halved_dx_reciprocal);

	apply_vorticity_kernel.setArg(0, temporary_p);
	apply_vorticity_kernel.setArg(1, w);
	apply_vorticity_kernel.setArg(2, temporary_w);
	apply_vorticity_kernel.setArg(3, halved_dx_reciprocal);
	apply_vorticity_kernel.setArg(4, time_step);
	apply_vorticity_kernel.setArg(5, vorticity_dx_scale);

	apply_gravity_kernel.setArg(0, temporary_w);
}

void Simulation::enqueueBoundaryKernel(cl::CommandQueue& cmd_queue, cl::Kernel& boundary_kernel) const
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

void Simulation::enqueueInnerKernel(cl::CommandQueue& cmd_queue, const cl::Kernel& kernel) const
{
	// Subtract 2 from the range in each dimension to account for 2 boundary cells
	// top & bottom or left & right cells
	const auto range  = cl::NDRange{workgroup_size, workgroup_size};

	for (uint y = 1; y < cell_count - 2; y += workgroup_size) {
		for (uint x = 1; x < cell_count - 2; x += workgroup_size) {
			cmd_queue.enqueueNDRangeKernel(kernel, cl::NDRange{x, y}, range);
		}
	}

	cmd_queue.enqueueBarrierWithWaitList();
}

void Simulation::calculate_advection()
{
	vector_advection_kernel.setArg(0, u);
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
	apply_impulse_kernel.setArg(0, w);
	apply_impulse_kernel.setArg(1, simulation_event.point);
	apply_impulse_kernel.setArg(2, simulation_event.value.as_vector);

	apply_impulse_kernel.setArg(3, Scalar{2});
	enqueueInnerKernel(cmd_queue, apply_impulse_kernel);
}

void Simulation::add_dye(const Event& simulation_event)
{
	add_dye_kernel.setArg(0, dye);
	add_dye_kernel.setArg(1, simulation_event.point);
	add_dye_kernel.setArg(2, simulation_event.value.as_scalar);
	add_dye_kernel.setArg(3, Scalar{64});
	enqueueInnerKernel(cmd_queue, add_dye_kernel);
}

void Simulation::apply_gravity()
{
	apply_gravity_kernel.setArg(0, w);
	enqueueInnerKernel(cmd_queue, apply_gravity_kernel);
}

void Simulation::apply_dye_boundary_conditions()
{
	dye_boundary_conditions_kernel.setArg(0, dye);
	enqueueBoundaryKernel(cmd_queue, dye_boundary_conditions_kernel);
}

void Simulation::apply_vorticity()
{
	vorticity_kernel.setArg(0, w);
	vorticity_kernel.setArg(1, temporary_p);
	enqueueInnerKernel(cmd_queue, vorticity_kernel);

	apply_vorticity_kernel.setArg(0, temporary_p);
	apply_vorticity_kernel.setArg(1, w);
	apply_vorticity_kernel.setArg(2, temporary_w);

	enqueueInnerKernel(cmd_queue, apply_vorticity_kernel);
	using std::swap;
	swap(w, temporary_w);
}

void Simulation::update()
{
	calculate_advection();
	auto events = events_from_ui->try_pop_all();
	if (not events.empty()) {
		cmd_queue.finish();
	}
	for (auto& simulation_event : events) {
		if (simulation_event.type == Event::Type::ADD_DYE) {
			add_dye(simulation_event);
		} else if (simulation_event.type == Event::Type::APPLY_FORCE) {
			apply_impulse(simulation_event);
		}
	}

	for (int i = 1; i < 10; ++ i) {
		Event imp_source;
		imp_source.point = Point{static_cast<cl_int>(i * 0.1 * cell_count), static_cast<cl_int>(cell_count * 0.8)};
		imp_source.value.as_vector = Vector{0, -20.0};
		apply_impulse(imp_source);
		imp_source.value.as_scalar = Scalar{0.01};
		add_dye(imp_source);
	}

	apply_gravity();

	apply_dye_boundary_conditions();

	apply_vector_boundary_conditions(w);

	advect_dye();

	apply_dye_boundary_conditions();
	calculate_diffusion();
	apply_vector_boundary_conditions(w);
	apply_vorticity();
	apply_vector_boundary_conditions(w);

	calculate_divergence_w();
	apply_scalar_boundary_conditions(divergence_w);

	calculate_p();
	calculate_gradient_p();

	calculate_u();
	apply_vector_boundary_conditions(u);

	ScalarField output_buffer;

	output_buffer.resize(total_cell_count);

	cl::copy(cmd_queue, dye, output_buffer.begin(), output_buffer.end());

	if (dye_buffers_wait_list.empty()) {
		to_ui->try_push(output_buffer);
	} else {
		dye_buffers_wait_list.emplace_back(output_buffer);
		to_ui->try_push_all(dye_buffers_wait_list);
	}
}
