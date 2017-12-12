#include <iostream>
#include "Functions.cpp"
#include "DirichletProblem.h"
#include <chrono>


int main(int argc, char * argv[])
{
	uint32_t n = 1000;
	bool bUseMPI = false;
	if (argc < 2)
	{
		std::cout << "Usage: task2.exe [num points in row] [-mpi]"<<std::endl;
		std::cout << "Num points in row sets by default n = 1000" << std::endl;
		std::cout << "Run without mpi" << std::endl;
	}
	else
	{
		n = atoi(argv[1]);
		bUseMPI = argc > 2 && *(argv[2]) == '-' ? true : false;
	}

	DirichletProblem<float> p(0.0, 0.0, 2.0, 2.0, n, n, FBoundaryFunc, FKnownFunc);
	p.SetPrecision(1e-4);
	double diff = 0.0;
auto start = std::chrono::system_clock::now();
	if (!bUseMPI)
	{
		diff = p.Solve();
	}
	else
	{
		diff = p.SolveMPI();
	}


auto end = std::chrono::system_clock::now();
auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//std::cout << "Time" << std::endl;
//std::cout << elapsed.count() << '\n';
	
	return 0;
}