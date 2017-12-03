#include <cmath>

inline float FBoundaryFunc(const float x, const  float y)
{
	return 1.0f + sin(x*y);
}

inline double DBoundaryFunc(const double x, const double y)
{
	return 1.0 + sin(x*y);
}

inline float FKnownFunc(const float x, const  float y)
{
	return (x*x+y*y)*sin(x*y);
}

inline double DKnownFunc(const double x, const double y)
{
	return (x*x + y*y)*sin(x*y);
}

inline float FBoundaryFunc10(const float x, const  float y)
{
	return sqrt(4+x*y);
}

inline float FKnownFunc10(const float x, const  float y)
{
	return (x*x+y*y)/(4*sqrt((4+x*y)*(4 + x*y)*(4 + x*y)));
}
