#include <iostream>
#include "Functions.cpp"
#include "DirichletProblem.h"

int main(int argc, char * argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: task2.exe <num points in row>"<<std::endl;
		return 0;
	}
	const uint32_t n = atoi(argv[1]);
	DirichletProblem<float> p(0.0, 0.0, 2.0, 2.0, n, n, FBoundaryFunc, FKnownFunc);
	p.SetPrecision(1e-4);
	//float diff = p.Solve();
	float diff = p.SolveMPI();
	//p.PrintSolution();
	return 0;
}