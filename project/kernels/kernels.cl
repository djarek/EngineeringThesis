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

typedef float2 Vector;
typedef Vector VectorField;
typedef float Scalar;
typedef int2 Point;

#define GlobalVectorField global Vector*
#define GlobalScalarField global Scalar*

inline size_t AT(size_t x, size_t y)
{
	return y*SIZE + x;
}

inline size_t AT_POS(Point pos)
{
	return AT(pos.x, pos.y);
}

inline Point getPosition()
{
	Point point = {get_global_id(0), get_global_id(1)};
	return point;
}

inline Scalar lerp_scalar(Scalar s, Scalar e, Scalar t)
{
	return s+(e-s)*t;
}

inline Scalar blerp_scalar(Scalar c00, Scalar c10, Scalar c01, Scalar c11, Scalar tx, Scalar ty){
	return lerp_scalar(lerp_scalar(c00, c10, tx), lerp_scalar(c01, c11, tx), ty);
}

inline Scalar bilinear_interpolation_scalar(const GlobalScalarField field, const Vector position)
{
	const int x = min(max((int)floor(position.x), 1), SIZE - 3);
	const int y = min(max((int)floor(position.y), 1), SIZE - 3);
	const int x1 = x;
	const int x2 = x + 1;
	const int y1 = y;
	const int y2 = y + 1;

	float x_pos = (fabs(position.x) - x1) / (x2 - x1);
	float y_pos = (fabs(position.y) - y1) / (y2 - y1);

	return fabs(blerp_scalar(field[AT(x1, y1)], field[AT(x2, y1)], field[AT(x1, y2)], field[AT(x2, y2)], x_pos, y_pos));
}

kernel void advect_scalar(const GlobalScalarField x, const GlobalVectorField u, GlobalScalarField x_out, const Scalar dx_reversed, const Scalar time_step, const Scalar dissipation)
{
	const Point position = getPosition();

	const Vector old_position = time_step * dx_reversed * u[AT_POS(position)];
	Vector vec_pos = {position.x, position.y};
	vec_pos -= old_position;

	x_out[AT_POS(position)] = bilinear_interpolation_scalar(x, vec_pos) * dissipation;
}

inline Vector lerp_vector(Vector s, Vector e, Scalar t)
{
	return s+(e-s)*t;
}

inline Vector blerp_vector(Vector c00, Vector c10, Vector c01, Vector c11, Scalar tx, Scalar ty){
	return lerp_vector(lerp_vector(c00, c10, tx), lerp_vector(c01, c11, tx), ty);
}

inline Vector bilinear_interpolation_vector(const GlobalVectorField field, const Vector position)
{
	const int x = min(max((int)floor(position.x), 1), SIZE - 3);
	const int y = min(max((int)floor(position.y), 1), SIZE - 3);
	const int x1 = x;
	const int x2 = x + 1;
	const int y1 = y;
	const int y2 = y + 1;

	float x_pos = (fabs(position.x) - x1) / (x2 - x1);
	float y_pos = (fabs(position.y) - y1) / (y2 - y1);

	return blerp_vector(field[AT(x1, y1)], field[AT(x2, y1)], field[AT(x1, y2)], field[AT(x2, y2)], x_pos, y_pos);
}

kernel void advect_vector(const GlobalVectorField x, const GlobalVectorField u, GlobalVectorField x_out, const Scalar dx_reversed, const Scalar time_step, const Vector dissipation)
{
	const Point position = getPosition();

	const Vector old_position = time_step * dx_reversed * u[AT_POS(position)];
	Vector vec_pos = {position.x, position.y};
	vec_pos -= old_position;

	x_out[AT_POS(position)] = bilinear_interpolation_vector(x, vec_pos) * dissipation;
}

kernel void vector_jacobi_iteration(const GlobalVectorField x, const GlobalVectorField b, GlobalVectorField x_out, const Scalar alpha, const Scalar beta_reciprocal)
{
	const Point position = getPosition();
	const int index = AT_POS(position);

	const Vector x_left = x[AT(position.x - 1, position.y)];
	const Vector x_right = x[AT(position.x + 1, position.y)];
	const Vector x_top = x[AT(position.x, position.y + 1)];
	const Vector x_bottom = x[AT(position.x, position.y - 1)];

	x_out[index] = (x_left + x_right + x_top + x_bottom + alpha * b[index]) * beta_reciprocal;
}

kernel void scalar_jacobi_iteration(const GlobalScalarField x, const GlobalScalarField b, GlobalScalarField x_out, const Scalar alpha, const Scalar beta_reciprocal)
{
	const Point position = getPosition();
	const int index = AT_POS(position);

	const Scalar x_left = x[AT(position.x - 1, position.y)];
	const Scalar x_right = x[AT(position.x + 1, position.y)];
	const Scalar x_top = x[AT(position.x, position.y + 1)];
	const Scalar x_bottom = x[AT(position.x, position.y - 1)];

	x_out[index] = (x_left + x_right + x_top + x_bottom + alpha * b[index]) * beta_reciprocal;
}

kernel void divergence(const GlobalVectorField w, GlobalScalarField divergence_w_out, const Scalar halved_reverse_dx)
{
	const Point position = getPosition();

	const Vector w_left = w[AT(position.x - 1, position.y)];
	const Vector w_right = w[AT(position.x + 1, position.y)];
	const Vector w_top = w[AT(position.x, position.y + 1)];
	const Vector w_bottom = w[AT(position.x, position.y - 1)];

	divergence_w_out[AT_POS(position)] = halved_reverse_dx * (w_right.x - w_left.x + w_top.y - w_bottom.y);

}

kernel void gradient(const GlobalScalarField p, GlobalVectorField gradient_p_out, const Scalar halved_reverse_dx)
{
	const Point position = getPosition();
	const int index = AT_POS(position);

	const Scalar p_left = p[AT(position.x - 1, position.y)];
	const Scalar p_right = p[AT(position.x + 1, position.y)];
	const Scalar p_top = p[AT(position.x, position.y + 1)];
	const Scalar p_bottom = p[AT(position.x, position.y - 1)];

	gradient_p_out[index].x = p_right - p_left;
	gradient_p_out[index].y = p_top - p_bottom;
}

kernel void subtract_gradient_p(const GlobalVectorField w, const GlobalVectorField gradient_p, GlobalVectorField u_out)
{
	const Point position = getPosition();
	const int index = AT_POS(position);

	u_out[index] = w[index] - gradient_p[index];
}

kernel void vector_boundary_condition(GlobalVectorField field, const Point offset)
{
	const Point position = getPosition();
	const Point position_offset = position + offset;

	field[AT_POS(position)] = -field[AT_POS(position_offset)];
}

kernel void scalar_boundary_condition(GlobalScalarField field, const Point offset)
{
	const Point position = getPosition();
	const Point position_offset = position + offset;

	field[AT_POS(position)] = field[AT_POS(position_offset)];
}

kernel void apply_impulse(GlobalVectorField w, const Point impulse_position, const Vector force, const Scalar impulse_range, const Scalar dt)
{
	const Point position = getPosition();

	int dist_from_impulse_squared = pown((Scalar)(position.x - impulse_position.x), 2) + pown((Scalar)(position.y - impulse_position.y), 2);

	w[AT_POS(position)] += 100 * force * dt * exp(-dist_from_impulse_squared / pown(impulse_range, 2));
}

kernel void apply_gravity(GlobalVectorField w)
{
	const Point position = getPosition();
	const Vector gravity = {0.0, 0.000};
	w[AT_POS(position)] += gravity;
}

kernel void add_dye(GlobalScalarField dye, const Point impulse_position, const Scalar dye_change, const Scalar impulse_range, const Scalar dt)
{
	const Point position = getPosition();

	int dist_from_impulse_squared = pown((Scalar)(position.x - impulse_position.x), 2) + pown((Scalar)(position.y - impulse_position.y), 2);

	dye[AT_POS(position)] += dye_change * dt * exp(-dist_from_impulse_squared / pown(impulse_range, 2));
}

kernel void apply_dye_boundary_conditions(GlobalScalarField dye, const Point offset)
{
	const Point position = getPosition();

	dye[AT_POS(position)] = 0.0;
}

kernel void vorticity(GlobalVectorField w, GlobalScalarField vorticity, Scalar halved_reverse_dx)
{
	const Point position = getPosition();
	const int index = AT_POS(position);

	const Vector w_left = w[AT(position.x - 1, position.y)];
	const Vector w_right = w[AT(position.x + 1, position.y)];
	const Vector w_top = w[AT(position.x, position.y + 1)];
	const Vector w_bottom = w[AT(position.x, position.y - 1)];

	vorticity[AT_POS(position)] = halved_reverse_dx * ((w_right.y - w_left.y) - (w_top.x - w_bottom.x));
}

static constant const Scalar EPSILON = 2.4414e-4; //2^-12

kernel void apply_voritcity_force(GlobalScalarField vorticity, GlobalVectorField w, GlobalVectorField w_out, Scalar halved_reverse_dx, Scalar time_step, Vector vorticity_dx_scale)
{

	const Point position = getPosition();
	const int index = AT_POS(position);

	const Scalar v_left = vorticity[AT(position.x - 1, position.y)];
	const Scalar v_right = vorticity[AT(position.x + 1, position.y)];
	const Scalar v_top = vorticity[AT(position.x, position.y + 1)];
	const Scalar v_bottom = vorticity[AT(position.x, position.y - 1)];

	const Scalar v_center = vorticity[index];

	Scalar force_x = fabs(v_top) - fabs(v_bottom);
	Scalar force_y = fabs(v_right) - fabs(v_left);

	Vector force = {force_x, force_y};
	Scalar mag_squared = max(EPSILON, dot(force, force));
	force *= rsqrt(mag_squared);

	force *= vorticity_dx_scale * v_center * (Vector)(1, -1);

	w_out[index] += time_step * force;
}
