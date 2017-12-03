#include <iostream>
#include "Functions.cpp"
#include "DirichletProblem.h"

int main(int argc, char * argv[])
{
	const uint32_t n = 1000;
	DirichletProblem<float> p(0.0, 0.0, 2.0, 2.0, n, n, FBoundaryFunc, FKnownFunc);
	p.SetPrecision(1e-4);
	//float diff = p.Solve();
	float diff = p.SolveMPI();
	//p.PrintSolution();
	return 0;
}