

template <typename VALTYPE>
Grid<VALTYPE>::Grid(const uint32_t nRows_, const uint32_t nCols_)
	: nRows(nRows_)
	, nCols(nCols_)
	, ptrData(new VALTYPE [nRows_*nCols_*sizeof(VALTYPE)])
{
	memset((VALTYPE*) ptrData.get(), VALTYPE(0), nRows_*nCols_ * sizeof(VALTYPE));
}


template <typename VALTYPE>
void Grid<VALTYPE>::operator -= (const Grid<VALTYPE> & grid)
{
	if (grid.nCols == nCols && grid.nRows == nRows)
	{
		VALTYPE * pData = ptrData.get();
		VALTYPE * pSub = grid.ptrData.get();
		pData += nCols + 1;//смещаем на елемент (1,1)
		pSub += nCols + 1;//смещаем на елемент (1,1)
		for (uint32_t y = 1; y < nRows - 1; ++y)
		{
			for (uint32_t x = 1; x < nCols - 1; ++x)
			{
				*pData -= *pSub;
				++pData;
				++pSub;
			}
			pData += 2;
			pSub += 2;
		}
	}
}

template <typename VALTYPE>
void Grid<VALTYPE>::operator *= (const VALTYPE nMultiplier)
{
	VALTYPE *pData = ptrData.get()+nCols+1;
	for (uint32_t y = 1; y < nRows-1; ++y)
	{
		for (uint32_t x = 1; x < nCols-1; ++x)
		{
			*pData *= nMultiplier;
			++pData;
		}
		pData += 2;
	}
}

template <typename VALTYPE>
void Grid<VALTYPE>::operator = (const Grid<VALTYPE> & grid)
{
	if (!ptrData.get())
	{
		ptrData = std::shared_ptr<VALTYPE>(new VALTYPE[grid.nCols*grid.nRows * sizeof(VALTYPE)]);
		nRows = grid.nRows;
		nCols = grid.nCols;
	}
	VALTYPE * pDst = ptrData.get();
	VALTYPE * pSrc = grid.ptrData.get();
	memcpy(pDst, pSrc, grid.nRows*grid.nCols * sizeof(VALTYPE));
	//for (uint64_t i = 0; i < grid.nRows*grid.nCols; ++i)
	//{
	//	*pDst = *pSrc;
	//	++pDst;
	//	++pSrc;
	//}
}

template <typename VALTYPE>
void Grid<VALTYPE>::Laplacian(const Grid<VALTYPE> & grid, const VALTYPE fStep, const VALTYPE fHalfStep)
{
	if (grid.nRows == nRows && grid.nCols == nCols && nRows > 2 && grid.nRows > 2)
	{
		auto * pPrevRowValue = grid.ptrData.get() + 1;
		auto * pCurRowValue = grid.ptrData.get() + nCols + 1;
		auto * pNextRowValue = grid.ptrData.get() + 2 * nCols + 1;
		auto * pCurDst = ptrData.get() + nCols + 1;

		for (uint32_t y = 1; y < nRows - 1; ++y)
		{
			for (uint32_t x = 1; x < nCols - 1; ++x)
			{
				auto fCurVal = *pCurRowValue;
				*pCurDst = (1.0 / fHalfStep) * ((fCurVal - *pPrevRowValue) / fStep -
					(*pNextRowValue - fCurVal) / fStep) +
					(1.0 / fHalfStep) *((fCurVal - *(pCurRowValue - 1)) / fStep -
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

template <typename VALTYPE>
VALTYPE Grid<VALTYPE>::MaxNormDifference()
{
	VALTYPE fMax = -std::numeric_limits<VALTYPE>::min();
	auto * pCurSrc = ptrData.get() + nCols + 1;
	for (uint32_t i = 1; i < nRows - 1; ++i)
	{
		for (uint32_t j = 1; j < nCols - 1; ++j)
		{
			auto fDiff = fabs(*pCurSrc);
			if (fDiff > fMax)
			{
				fMax = fDiff;
			}
			++pCurSrc;
		}
		pCurSrc += 2;
	}
	return fMax;
}

template <typename VALTYPE>
void Grid<VALTYPE>::Fill(std::function<VALTYPE(VALTYPE, VALTYPE)> Func_, FillTypes fillType,
						const VALTYPE x0, const VALTYPE y0, const VALTYPE fStep)
{
	VALTYPE * pValues = ptrData.get();
	if (fillType == FILL_BOUNDARY)
	{
		VALTYPE x1 = (nCols-1)*fStep + x0;
		VALTYPE y1 = (nRows-1)*fStep + y0;
		memset(pValues, VALTYPE(0), nRows * nCols * sizeof(VALTYPE));
		for (uint32_t y = 0; y < nRows; ++y)
		{
			pValues[y*nCols] = Func_(x0, y0+y*fStep);
			pValues[(y + 1)*nCols - 1] = Func_(x1, y0 + y*fStep);
		}
		for (uint32_t x = 0; x < nCols; ++x)
		{
			pValues[x] = Func_(x0+x*fStep, y0);
			pValues[(nRows - 1)*nCols + x] = Func_(x0+x*fStep, y1);
		}
	}
	else
	{
		auto * pCurValue = pValues;
		for (uint32_t y = 0; y < nRows; ++y)
		{
			for (uint32_t x = 0; x < nCols; ++x)
			{
				*pCurValue = Func_(x0+x*fStep,y0+y*fStep);
				pCurValue++;
			}
		}
	}
}



template <typename VALTYPE>
GridStorage<VALTYPE>::GridStorage(const uint64_t nSize, const uint32_t nNumParts)
	: nSize(nSize_)
{
	pAllMem = std::shared_ptr<VALTYPE>(new VALTYPE [nSize*nNumParts * sizeof(VALTYPE)+1]);
	uint64_t nPartBegin = 0;
	vPartPointers.resize(nNumParts);
	for (auto & it : vPartPointers)
	{
		it.p = pAllMem.get()+nPartBegin;
		it.bIsFree = true;
		nPartBegin += nSize;
	}
}

template <typename VALTYPE>
std::shared_ptr<VALTYPE> GridStorage<VALTYPE>::getMem()
{
	for (auto & it : vPartPointers)
	{
		if (it.bIsFree)
		{
			it.bIsFree = false;
			return std::shared_ptr(it.p,Deleter());
		}
	}
	return 0;
}

template <typename VALTYPE>
void  GridStorage<VALTYPE>::free(std::shared_ptr<VALTYPE>& ptr)
{
	for (auto & it : vPartPointers)
	{
		if (it.p == pPointer)
		{
			it.bIsFree = true;
			memset(it.p.get(), VALTYPE(0), nSize * sizeof(VALTYPE));
			return;
		}
	}
}

template<typename VALTYPE>
VALTYPE DotProduct(const Grid<VALTYPE> & grid1, const Grid<VALTYPE> & grid2, const VALTYPE k)
{
	VALTYPE fResult = 0;
	const VALTYPE fPowK = k * k;
	auto * pCurMult1 = grid1.ptrData.get()+grid1.nCols+1;
	auto * pCurMult2 = grid2.ptrData.get()+grid2.nCols+1;
	for (uint32_t y = 1; y < grid1.nRows-1; ++y)
	{
		for (uint32_t x = 1; x < grid1.nCols-1; ++x)
		{
			fResult += fPowK * (*pCurMult1) * (*pCurMult2);
			++pCurMult1;
			++pCurMult2;
		}
			pCurMult1+=2;
			pCurMult2+=2;
	}
	return fResult;
}