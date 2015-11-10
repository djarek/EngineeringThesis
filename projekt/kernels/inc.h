typedef float2 Vector;

typedef Vector VectorField;
typedef float Scalar;
typedef int2 Point;


#define GlobalVectorField global Vector*
#define GlobalScalarField global Scalar*

// #define GlobalVectorField Vector*
// #define GlobalScalarField Scalar*

#define SIZE 4
#define AT(x, y) y*SIZE + x
#define AT_POS(pos) AT(pos.x, pos.y)
