#ifndef DIRICHLET_PROBLEM_H
#define DIRICHLET_PROBLEM_H

#include "GridOperations.h"
#include<iostream>
#include <iomanip>

template <typename VALTYPE>
class DirichletProblem
{
public:

	DirichletProblem()
		: nPointsX(0)
		, nPointsY(0)
		, fStep(0)
		, fHalfStep(0)
		, x0(0)
		, x1(1)
		, y0(0)
		, y1(1)
		, eps(1e-6)
	{}
#ifdef std11
	DirichletProblem(const VALTYPE x0_, const VALTYPE y0_,
		const VALTYPE x1_, const VALTYPE y1_,
		const uint32_t nPointsX_, const uint32_t nPointsY_,
		const std::function<VALTYPE(const VALTYPE, const VALTYPE)> BoundaryFunc_,
		const std::function<VALTYPE(const VALTYPE, const VALTYPE)> AppFunc_);
#else
	DirichletProblem(const VALTYPE x0_, const VALTYPE y0_,
		const VALTYPE x1_, const VALTYPE y1_,
		const uint32_t nPointsX_, const uint32_t nPointsY_,
		VALTYPE (* BoundaryFunc_)(const VALTYPE, const VALTYPE),
		VALTYPE (* AppFunc_)(const VALTYPE, const VALTYPE));
#endif
	VALTYPE Solve();
	VALTYPE SolveMPI();
	void InitSubproblem_mpi();
	void AgregateSolution();

	void SetPrecision(const VALTYPE eps_) { eps = eps_;}
	void PrintSolution()
	{
		VALTYPE * data = solution.ptrData.get();
		if (data)
		{
			double maxV = -std::numeric_limits<VALTYPE>::min();
			double sum = 0.0;
			int32_t numPoints = 0;
			std::cout << "MaxNorm Error: " << error << std::endl;
			std::cout << "Iterations num: " << num_iter << std::endl;
			for (int i = 1; i < solution.nRows-1; ++i)
			{
				for (int j = 1; j < solution.nCols-1; ++j)
				{
					//std::cout << data[i*solution.nCols + j] << ' ';
					double diff = fabs(data[i*solution.nCols + j] - BoundaryFunction(x0+j*fStepX,y0+i*fStepX));
					if (!(i == 0 || j == 0 || (i == solution.nRows - 1) || (j == solution.nCols-1)))
					{
						sum += diff;
						++numPoints;
					}
					if (diff > maxV)
					{
						maxV = diff;
					}
				}
				//std::cout << std::endl;
			}
			//std::cout<<std::endl;
			std::cout << "Max Error: " << maxV << std::endl;
			std::cout << "Average error: " << sum / (numPoints) << std::endl;

		}
	}

	~DirichletProblem() {}
	

	uint32_t nPointsX;
	uint32_t nPointsY;
	VALTYPE x0;
	VALTYPE x1;
	VALTYPE y0;
	VALTYPE y1;
private:
	VALTYPE fStepX;
	VALTYPE fStepY;
	VALTYPE fHalfStepX;
	VALTYPE fHalfStepY;
	VALTYPE fAvStepX;
	VALTYPE fAvStepY;
	uint32_t nPointsX_nompi;
	uint32_t nPointsY_nompi;
	VALTYPE x0_nompi;
	VALTYPE x1_nompi;
	VALTYPE y0_nompi;
	VALTYPE y1_nompi;
	VALTYPE fStepX_nompi;
	VALTYPE fStepY_nompi;
	VALTYPE fAvStepX_nompi;
	VALTYPE fAvStepY_nompi;
	VALTYPE eps;
#ifdef std11
	std::function<VALTYPE(const VALTYPE, const VALTYPE)> BoundaryFunction;
	std::function<VALTYPE(const VALTYPE, const VALTYPE)> AppFunction;
#else
	VALTYPE(*BoundaryFunction)(const VALTYPE, const VALTYPE);
	VALTYPE(*AppFunction)(const VALTYPE, const VALTYPE);
#endif
	Grid<VALTYPE> solution;
	//GridMPI<VALTYPE> solutionMPI;
	uint32_t num_iter;
	VALTYPE error;
	/*
	BoundaryBuffer<VALTYPE> BBufer;
	BoundaryBuffer<VALTYPE> SendBufer;
	ProcessorNeighborhood Neighbors;
	*/
	MPI_Comm Comm_mpi;
	int nProcIndex;
	int nNumProc;
};


#include "DirichletProblem.hpp"
#endif