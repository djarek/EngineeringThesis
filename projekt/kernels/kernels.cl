
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

inline Vector lerp(Vector s, Vector e, float t)
{
	return s+(e-s)*t;
}

inline Vector blerp(Vector c00, Vector c10, Vector c01, Vector c11, float tx, float ty){
	return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

inline Vector bilinear_interpolation(const GlobalVectorField field, const Vector position)
{
	const int x = max((int)floor(position.x), 0);
	const int y = max((int)floor(position.y), 0);
	const int x1 = min(x, SIZE - 1);
	const int x2 = min(x + 1, SIZE - 1);
	const int y1 = min(y, SIZE - 1);
	const int y2 = min(y + 1, SIZE - 1);

	return blerp(field[AT(x1, y1)], field[AT(x2, y1)], field[AT(x1, y2)], field[AT(x2, y2)], position.x, position.y);
}

kernel void advect(const GlobalVectorField x, const GlobalVectorField u, GlobalVectorField x_out, const float dx_reversed, const float time_step, const Vector dissipation)
{
	const Point position = getPosition();

	const Vector old_position = time_step * dx_reversed * u[AT_POS(position)];
	Vector vec_pos = {position.x, position.y};
	vec_pos -= old_position;

	x_out[AT_POS(position)] += bilinear_interpolation(x, vec_pos) * dissipation;
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
