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


inline float FBoundaryFuncAzat(const float x, const  float y)
{
	return (1.0-x*x)*(1.0 - x*x)+ (1.0 - y*y)*(1.0 - y*y);
}

inline float FKnownFuncAzat(const float x, const  float y)
{
	return 4*(2-3*x*x-3*y*y);
}
