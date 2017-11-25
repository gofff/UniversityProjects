//#include "MathCore.h"

#ifndef MATHCORE_HPP
#define MATHCORE_HPP

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>

template <typename VALTYPE>
GridStorage<VALTYPE>::GridStorage(const uint64_t nSize_, const uint8_t nNumParts)
{
	pMem = new VALTYPE[nNumParts * nSize_ * sizeof(VALTYPE)];
	memset(pMem, VALTYPE(0), nNumParts * nSize_ * sizeof(VALTYPE));
	nSize = nSize_;
	vPartPointers.resize(nNumParts);
	VALTYPE * pCurMem = pMem;
	for (auto & it : vPartPointers)
	{
		it.p = pCurMem;
		it.bIsFree = true;
		pCurMem += nSize;
	}
}


template <typename VALTYPE>
GridStorage<VALTYPE>::~GridStorage()
{
	//TODO: check this mem
	delete[] pMem;
}

template <typename VALTYPE>
VALTYPE * GridStorage<VALTYPE>::getMem()
{
	for (auto & it : vPartPointers)
	{
		if (it.bIsFree)
		{
			it.bIsFree = false;
			return it.p;
		}
	}
	return nullptr;
}

template <typename VALTYPE>
void GridStorage<VALTYPE>::free(VALTYPE * pPointer)
{
	for (auto & it : vPartPointers)
	{
		if (it.p == pPointer)
		{
			it.bIsFree = true;
			memset(it.p, VALTYPE(0), nSize*sizeof(VALTYPE));
			return;
		}
	}
}

template <typename VALTYPE>
DiscreteFunction<VALTYPE>::DiscreteFunction(const float x0, const float y0,
											const float x1, const float y1,
											const uint32_t nRows_, const uint32_t nCols_,
											std::function<VALTYPE(const VALTYPE, const VALTYPE)> Func,
											FunctionTypes functionType)

	:nRows(nRows_)
	,nCols(nCols_)
	,segmentX(x0,x1)
	,segmentY(y0,y1)
{
	pValues = new VALTYPE[nRows_ * nCols_ * sizeof(VALTYPE)];
	fStep = (x1 - x0) / (nCols-1);
	fHalfStep = fStep / 2;
	
	if (functionType == FT_KNOWN)
	{
		auto * pCurValue = pValues;
		for (uint32_t y = 0; y < nRows; ++y)
		{
			for (uint32_t x = 0; x < nCols; ++x)
			{
				*pCurValue = Func(x*fStep, y*fStep);
				pCurValue++;
			}
		}
	}
	else
	{
		memset(pValues, VALTYPE(0), nRows_ * nCols_ * sizeof(VALTYPE));
		for (uint32_t y = 0; y < nRows; ++y)
		{
			pValues[y*nCols] = Func(x0, y*fStep);
			pValues[(y+1)*nCols-1] = Func(x1, y*fStep);
		}
		for (uint32_t x = 0; x < nCols; ++x)
		{
			pValues[x] = Func(x*fStep, y0);
			pValues[(nRows - 1)*nCols + x] = Func(x*fStep, y1);
		}
	}

}

template <typename VALTYPE>
DiscreteFunction<VALTYPE>::~DiscreteFunction()
{
	if (pValues)
	{
		delete[] pValues;
	}
}

template <typename VALTYPE>
void DiscreteFunction<VALTYPE>::EstimateSimpleDiscrep(const VALTYPE * pSrc, const DiscreteFunction & F, VALTYPE * pDst)
{
	//TODO: make error if sizes don't match
	Laplacian(pSrc, pDst, nRows, nCols, fStep, fHalfStep);
	Subtract(pDst, F.pValues, pDst, nRows,nCols);
}

template <typename VALTYPE>
void DiscreteFunction<VALTYPE>::SolveDirichletProblem(const DiscreteFunction & F,
	std::function<VALTYPE(VALTYPE, VALTYPE)> boundaryFunction)
{
	const VALTYPE eps = 0.0000001; //сделай нормально
	GridStorage<VALTYPE> storage(nRows*nCols, 3);
	VALTYPE * pDiscrep = storage.getMem();
	VALTYPE * pDiscrepLap = storage.getMem();
	VALTYPE * pDiscrepPrev = storage.getMem();
	uint32_t nNumIter = 0;
	VALTYPE diff = MaxNormDifference(pDiscrepPrev, pDiscrep, nRows, nCols);
	while (nNumIter<2 || diff > eps)
	{
		std::cout << std::fixed << std::setprecision(8);
		std::cout << nNumIter << ": " << diff<<::std::endl;
		storage.free(pDiscrepPrev);
		pDiscrepPrev = pDiscrep;
		for (uint32_t y = 0; y < nRows; ++y)
		{
			pDiscrepPrev[y*nCols] = pValues[y*nCols];
			pDiscrepPrev[(y + 1)*nCols - 1] = pValues[(y + 1)*nCols - 1];
		}
		for (uint32_t x = 0; x < nCols; ++x)
		{
			pDiscrepPrev[x] = pValues[x];
			pDiscrepPrev[(nRows - 1)*nCols + x] = pValues[(nRows - 1)*nCols + x];
		}
		pDiscrep = storage.getMem();

		EstimateSimpleDiscrep(nNumIter?pDiscrepPrev:pValues, F, pDiscrep);
		Laplacian(pDiscrep, pDiscrepLap, nRows, nCols, fStep, fHalfStep);
		VALTYPE tau = DotProduct(pDiscrep, pDiscrep, nRows, nCols, fHalfStep) / 
						DotProduct(pDiscrepLap,pDiscrep,nRows,nCols,fHalfStep);
		Mult(pDiscrep, tau,nRows, nCols);
		Subtract(nNumIter ? pDiscrepPrev : pValues, pDiscrep, pDiscrep, nRows,nCols);
		++nNumIter;
		diff = MaxNormDifference(pDiscrepPrev, pDiscrep, nRows, nCols);
		
	}
	
}



template <typename VALTYPE>
void Laplacian(const VALTYPE * pSrc, VALTYPE * pDst,
				const uint32_t nRows, const uint32_t nCols,
				const VALTYPE fStep, const VALTYPE fHalfStep)
{
	if (nRows > 2)
	{
		auto * pPrevRowValue = pSrc + 1;
		auto * pCurRowValue = pSrc + nCols + 1;
		auto * pNextRowValue = pSrc + 2 * nCols + 1;
		auto * pCurDst = pDst + nCols + 1;

		for (uint32_t y = 1; y < nRows - 1; ++y)
		{
			for (uint32_t x = 1; x < nCols - 1; ++x)
			{
				auto fCurVal = *pCurRowValue;
				*pCurDst = (1.0 / fHalfStep) * ((fCurVal - *pPrevRowValue) / fStep -
												(*pNextRowValue-fCurVal) / fStep) +
							(1.0 / fHalfStep) *((fCurVal - *(pCurRowValue-1)) / fStep -
												(*(pCurRowValue + 1) - fCurVal) / fStep);
				++pCurDst;
				++pPrevRowValue;
				++pCurRowValue;
				++pNextRowValue;
			}
			pCurDst += 2;
			pPrevRowValue += 2;
			pCurRowValue += 2;
			pNextRowValue += 2;
		}
	}
}

template<typename VALTYPE>
VALTYPE DotProduct(const VALTYPE * pMult1, const VALTYPE * pMult2,
					const uint32_t nRows, const uint32_t nCols, const VALTYPE fHalfStep)
{
	VALTYPE fResult = 0;
	uint64_t nSize = nRows * nCols;
	const VALTYPE fPowHalfStep = fHalfStep * fHalfStep;
	auto * pCurMult1 = pMult1;
	auto * pCurMult2 = pMult2;
	for (uint64_t i = 0; i < nSize; ++i)
	{
		fResult += fPowHalfStep * (*pCurMult1) * (*pCurMult2);
		++pCurMult1;
		++pCurMult2;
	}
	return fResult;
}


template<typename VALTYPE>
VALTYPE MaxNormDifference(const VALTYPE * pSrc1, const VALTYPE * pSrc2,
	const uint32_t nRows, const uint32_t nCols)
{
	VALTYPE fMax = -std::numeric_limits<VALTYPE>::min();
	const uint64_t nSize = nRows*nCols;
	auto * pCurSrc1 = pSrc1+nCols+1;
	auto * pCurSrc2 = pSrc2+nCols+1;
	for (uint32_t i = 1; i < nRows - 1; ++i)
	{
		for (uint32_t j = 1; j < nCols - 1; ++j)
		{
			auto fDiff = fabs(*pCurSrc1 - *pCurSrc2);
			if (fDiff > fMax)
			{
				fMax = fDiff;
			}
			++pCurSrc1;
			++pCurSrc2;
		}
		pCurSrc1+=2;
		pCurSrc2+=2;
	}
	return fMax;
}

//mem1 = mem1 - mem2
template<typename VALTYPE>
inline void Subtract(const VALTYPE * pMem1, const VALTYPE * pMem2, VALTYPE * pDst, const uint32_t nRows, const uint32_t nCols)
{
	for (uint32_t i = 1; i < nRows-1; ++i)
	{
		for (uint32_t j = 1; j < nCols-1; ++j)
		{
			pDst[i*nCols+j] = pMem1[i*nCols + j] - pMem2[i*nCols + j];
		}
	}
}

template<typename VALTYPE>
inline void Mult(VALTYPE * pMem1, const VALTYPE val, const uint32_t nRows, const uint32_t nCols)
{
	for (uint32_t i = 1; i < nRows-1; ++i)
	{
		for (uint32_t j = 1; j < nCols-1; ++j)
		{
			pMem1[i*nCols + j] *= val;
		}
	}
}
#endif