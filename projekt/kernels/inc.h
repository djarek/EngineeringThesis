typedef float2 Vector;

typedef Vector VectorField;
typedef float Scalar;
typedef int2 Point;


#define GlobalVectorField global Vector*
#define GlobalScalarField global Scalar*

// #define GlobalVectorField Vector*
// #define GlobalScalarField Scalar*

inline size_t AT(size_t x, size_t y)
{
	return y*SIZE + x;
}

inline size_t AT_POS(Point pos)
{
	return AT(pos.x, pos.y);
}