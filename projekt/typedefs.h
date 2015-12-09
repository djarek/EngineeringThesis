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

#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <CL/opencl.h>

using Scalar = cl_float;
using Vector = cl_float2;
using Point  = cl_int2;
using Offset = Point;
using ScalarField = std::vector<Scalar>;
using VectorField = std::vector<Vector>;

struct Event {
	enum class Type {
		ADD_DYE,
		APPLY_FORCE
	};

	Type type {Type::ADD_DYE};
	Point point;
	union {
		Scalar as_scalar;
		Vector as_vector;
	} value;
};

#endif //TYPEDEFS_H
