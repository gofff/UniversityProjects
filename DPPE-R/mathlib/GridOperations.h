#ifndef GRID_OPERATIONS_H
#define GRID_OPERATIONS_H

//#define LOG
#define OMP
//#define std11

#include <deque>
#include <limits>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "mpi.h"
#ifdef std11
#include <functional>
#endif

enum FillTypes {FILL_ALL, FILL_BOUNDARY};

template <typename VALTYPE>
class Grid
{
public:
	VALTYPE* ptrData;
	int nRows;
	int nCols;

	Grid()
	:ptrData(0)
	,nRows(0)
	,nCols(0)
	{}
	Grid(const int nRows_, const int nCols_);
	virtual ~Grid();

	void operator -= (const Grid<VALTYPE> & grid);
	void operator *= (const VALTYPE nMultiplier);
	virtual void operator = (const Grid<VALTYPE> & grid);

	void Laplacian(const Grid & grid, const VALTYPE& fStepX, const VALTYPE& fStepY, 
					const VALTYPE& fHalfStepX, const VALTYPE& fHalfStepY);
	VALTYPE MaxNormDifference() const;
	void dump(const std::string str)
	{
#ifdef LOG
		std::cout << str << std::endl;
		std::cout << std::fixed << std::setprecision(4);
		for (int i = 0; i < nRows; ++i)
		{
			for (int j = 0; j < nCols; ++j)
			{
				std::cout << ptrData[i*nCols + j] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
#endif
	}
#ifdef std11
	virtual void Fill(std::function<VALTYPE(VALTYPE, VALTYPE)> Func_, FillTypes fillType,
				const VALTYPE x0, const VALTYPE y0, const VALTYPE fStepX, const VALTYPE fStepY);
#else
	virtual void Fill(VALTYPE (*Func_)(VALTYPE, VALTYPE), FillTypes fillType,
		const VALTYPE x0, const VALTYPE y0, const VALTYPE fStepX, const VALTYPE fStepY);
#endif
};

template <typename VALTYPE>
class GridMPI : public Grid<VALTYPE>
{
public:

	int neighbors[4];
	int nProcIndex;
	VALTYPE * pTopBuf;
	VALTYPE * pBotBuf;
	VALTYPE * pLeftBuf;
	VALTYPE * pRightBuf;
	
	void operator = (const GridMPI<VALTYPE> & grid);

	GridMPI(const int nRows_, const int nCols_, const int nProcIndex_, const int nNumProc);
	~GridMPI();

	bool hasTopNeighb() const { return neighbors[3] > -1; }
	bool hasBotNeighb() const { return neighbors[2] > -1; }
	bool hasLeftNeighb() const { return neighbors[1] > -1; }
	bool hasRightNeighb() const { return neighbors[0] > -1; }

	void Send(const int tag, const MPI_Comm & comm);
	void Refresh(const int tag, const MPI_Comm & comm);


#ifdef std11
	void Fill(std::function<VALTYPE(VALTYPE, VALTYPE)> Func_, FillTypes fillType,
		const VALTYPE x0, const VALTYPE y0, const VALTYPE fStepX, const VALTYPE fStepY);
#else
	void Fill(VALTYPE(*Func_)(VALTYPE, VALTYPE), FillTypes fillType,
		const VALTYPE x0, const VALTYPE y0, const VALTYPE fStepX, const VALTYPE fStepY);
#endif
};

/*
template<typename VALTYPE>
class GridStorage
{
public:
	GridStorage(const uint64_t nSize, const int nNumParts);
	~GridStorage() {};
	std::shared_ptr<VALTYPE> getMem();
	void free(std::shared_ptr<VALTYPE>& ptr);
private:
	std::shared_ptr<VALTYPE> pAllMem;
	struct SmartPointer
	{
		VALTYPE * p;
		bool bIsFree;
	};
	struct Deleter
	{
		void operator() (VALTYPE * p) const { return; };
	};
	std::deque<SmartPointer> pointerQueue;
	uint64_t nSize;
};
*/

#include "GridOperations.hpp"
#endif