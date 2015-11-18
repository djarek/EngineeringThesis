#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <CL/opencl.h>

using Scalar = cl_float;
using Vector = cl_float2;
using Point  = cl_int2;
using Offset = Point;
using ScalarField = std::vector<Scalar>;
using VectorField = std::vector<Vector>;

#endif //TYPEDEFS_H
