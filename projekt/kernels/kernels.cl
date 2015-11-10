#include "kernels/inc.h"

Point getPosition()
{
	Point point = {get_global_id(0), get_global_id(1)};
	return point;
}

Vector bilinear_interpolation(const GlobalVectorField field, const Vector position)
{
	const float x = position.x;
	const float y = position.y;
	
	const int x1 = floor(position.x);
	const int x2 = ceil(position.x);
	const int y1 = floor(position.y);
	const int y2 = ceil(position.y);

	Vector ret = field[AT(x1, y1)] * (x2 - x) * (y2 - y);
	ret += field[AT(x2, y1)] * (x - x1) * (y2 - y);
	ret += field[AT(x1, y2)] * (x2 - x) * (y - y1);
	ret += field[AT(x2, y2)] * (x - x1) * (y - y1);
	ret /= (x2 - x1) * (y2 - y1);

	return ret;
}

kernel void advect(const GlobalVectorField x, const GlobalVectorField u, GlobalVectorField x_out, const float dx_reversed, const float time_step)
{
	const Point position = getPosition();

	const Vector old_position = time_step * dx_reversed * u[AT_POS(position)];
	Vector vec_pos = {position.x, position.y};
	vec_pos -= old_position;

	x_out[AT_POS(position)] = bilinear_interpolation(x, vec_pos);
}

kernel void vector_jacobi_iteration(const GlobalVectorField x, const GlobalVectorField b, GlobalVectorField x_out, const float alpha, const float beta_reciprocal)
{
	const Point position = getPosition();
	const int index = AT_POS(position);

	const Vector x_left = x[AT(position.x - 1, position.y)];
	const Vector x_right = x[AT(position.x + 1, position.y)];
	const Vector x_top = x[AT(position.x, position.y + 1)];
	const Vector x_bottom = x[AT(position.x, position.y - 1)];

	x_out[index] = (x_left + x_right + x_top + x_bottom + alpha * b[index]) * beta_reciprocal;
}

kernel void scalar_jacobi_iteration(const GlobalScalarField x, const GlobalScalarField b, GlobalScalarField x_out, const float alpha, const float beta_reciprocal)
{
	const Point position = getPosition();
	const int index = AT_POS(position);

	const Scalar x_left = x[AT(position.x - 1, position.y)];
	const Scalar x_right = x[AT(position.x + 1, position.y)];
	const Scalar x_top = x[AT(position.x, position.y + 1)];
	const Scalar x_bottom = x[AT(position.x, position.y - 1)];

	x_out[index] = (x_left + x_right + x_top + x_bottom + alpha * b[index]) * beta_reciprocal;
}

kernel void divergence(const GlobalVectorField w, GlobalScalarField divergence_w_out, const float halved_reverse_dx)
{
	const Point position = getPosition();

	const Vector w_left = w[AT(position.x - 1, position.y)];
	const Vector w_right = w[AT(position.x + 1, position.y)];
	const Vector w_top = w[AT(position.x, position.y + 1)];
	const Vector w_bottom = w[AT(position.x, position.y - 1)];

	divergence_w_out[AT_POS(position)] = halved_reverse_dx * (w_right.x - w_left.x + w_top.y - w_bottom.y);
}

kernel void gradient(const GlobalScalarField p, GlobalVectorField gradient_p_out, const float halved_reverse_dx)
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