#include <iostream>
#include "Functions.cpp"
#include "DirichletProblem.h"
#include <chrono>


int main(int argc, char * argv[])
{
	uint32_t n = 1000;
	if (argc < 2)
	{
		std::cout << "Usage: task2.exe <num points in row>"<<std::endl;
		std::cout << "Num points in row sets by default n = 1000" << std::endl;
	}
	else
	{
		n = atoi(argv[1]);
	}

	DirichletProblem<float> p(0.0, 0.0, 2.0, 2.0, n, n, FBoundaryFunc, FKnownFunc);
	p.SetPrecision(1e-4);

auto start = std::chrono::system_clock::now();

	float diff = p.Solve();
	//float diff = p.SolveMPI();


auto end = std::chrono::system_clock::now();
auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "Time" << std::endl;
std::cout << elapsed.count() << '\n';
	

	return 0;
}