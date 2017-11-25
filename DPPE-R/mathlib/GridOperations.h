#ifndef GRID_OPERATIONS_H
#define GRID_OPERATIONS_H

#include <memory>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include<iostream>
#include <iomanip>
enum FillTypes {FILL_ALL, FILL_BOUNDARY};

template <typename VALTYPE>
class Grid
{
public:
	std::shared_ptr<VALTYPE> ptrData;
	uint32_t nRows;
	uint32_t nCols;

	Grid() = default;
	Grid(const uint32_t nRows_, const uint32_t nCols_);

	void operator -= (const Grid<VALTYPE> & grid);
	void operator *= (const VALTYPE nMultiplier);
	void operator = (const Grid<VALTYPE> & grid);

	void Laplacian(const Grid & grid, const VALTYPE fStep, const VALTYPE fHalfStep);
	VALTYPE MaxNormDifference();
	void dump(const std::string str)
	{
#ifdef LOG
		std::cout << str << std::endl;
		std::cout << std::fixed << std::setprecision(4);
		for (int i = 0; i < nRows; ++i)
		{
			for (int j = 0; j < nCols; ++j)
			{
				std::cout << ptrData.get()[i*nCols + j] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
#endif
	}
	void Fill(std::function<VALTYPE(VALTYPE, VALTYPE)> Func_, FillTypes fillType,
				const VALTYPE x0, const VALTYPE y0, const VALTYPE fStep);
private:
	Grid(Grid<VALTYPE>& grid) = default;
	Grid(Grid<VALTYPE>&&) = default;
};


template<typename VALTYPE>
class GridStorage
{
public:
	GridStorage(const uint64_t nSize, const uint32_t nNumParts);
	~GridStorage();
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


#include "GridOperations.hpp"
#endif