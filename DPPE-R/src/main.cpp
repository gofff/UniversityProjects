#include <iostream>
//#include "MathCore.h"
#include "Functions.cpp"
#include "DirichletProblem.h"
#include <Windows.h>

int main(int argc, char * argv[])
{
LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
LARGE_INTEGER Frequency;

QueryPerformanceFrequency(&Frequency);
QueryPerformanceCounter(&StartingTime);

	const uint32_t n = 1000;
	DirichletProblem<double> p(-1.0f, -1.0f, 1.0f, 1.0f, n, n, FBoundaryFuncAzat, FKnownFuncAzat);
	p.SetPrecision(1e-4);
	double diff = p.Solve();

QueryPerformanceCounter(&EndingTime);
ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
ElapsedMicroseconds.QuadPart *= 1000;
ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
std::cout << "TIME: " << ElapsedMicroseconds.QuadPart << std::endl;

	p.PrintSolution();
	getchar();
	return 0;
}