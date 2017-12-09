#include "mpi.h"

template <typename VALTYPE>
Grid<VALTYPE>::Grid(const int nRows_, const int nCols_)
	: nRows(nRows_)
	, nCols(nCols_)
{
	ptrData = new VALTYPE[nRows_ * nCols_ * sizeof(VALTYPE)];
	memset(ptrData, VALTYPE(0), nRows_ * nCols_ * sizeof(VALTYPE));
}

template <typename VALTYPE>
Grid<VALTYPE>::~Grid()
{
	if (ptrData)
	{
		delete[] ptrData;
	}
}


template <typename VALTYPE>
void Grid<VALTYPE>::operator -= (const Grid<VALTYPE> & grid)
{
	if (grid.nCols != nCols || grid.nRows != nRows)
	{
		return;
	}
	VALTYPE * pData = ptrData;
	VALTYPE * pSub = grid.ptrData;
#ifdef std11
	pData += nCols + 1;//смещаем на елемент (1,1)
	pSub += nCols + 1;//смещаем на елемент (1,1)
	for (int y = 1; y < nRows - 1; ++y)
	{
		for (int x = 1; x < nCols - 1; ++x)
		{
			*pData -= *pSub;
			++pData;
			++pSub;
		}
		pData += 2;
		pSub += 2;
	}
#else

#ifdef OMP
	#pragma omp parallel for collapse(2)
#endif
	for (int y = 1; y < nRows - 1; ++y)
	{
		for (int x = 1; x < nCols - 1; ++x)
		{
			pData[y*nCols + x] -= pSub[y*nCols + x];
		}
	}
#endif
}

template <typename VALTYPE>
void Grid<VALTYPE>::operator *= (const VALTYPE nMultiplier)
{
#ifdef std11
	VALTYPE *pData = ptrData + nCols + 1;

	for (int y = 1; y < nRows-1; ++y)
	{
		for (int x = 1; x < nCols-1; ++x)
		{
			*pData *= nMultiplier;
			++pData;
		}
		pData += 2;
	}
#else
	VALTYPE *pData = ptrData;
#ifdef OMP
	#pragma omp parallel for collapse(2)
#endif
	for (int y = 1; y < nRows - 1; ++y)
	{
		for (int x = 1; x < nCols - 1; ++x)
		{
			pData[y*nCols + x] *= nMultiplier;
		}
	}
#endif
}

template <typename VALTYPE>
void Grid<VALTYPE>::operator = (const Grid<VALTYPE> & grid)
{
	if (ptrData && (nRows * nCols) != grid.nRows * grid.nCols)
	{
		delete[] ptrData;
		ptrData = 0;
	}
	else if ((nRows * nCols) != grid.nRows * grid.nCols)
	{
		ptrData = new VALTYPE[grid.nCols * grid.nRows * sizeof(VALTYPE)];
	}

	nRows = grid.nRows;
	nCols = grid.nCols;
	VALTYPE * pDst = ptrData;
	VALTYPE * pSrc = grid.ptrData;
	memcpy(pDst, pSrc, grid.nRows * grid.nCols * sizeof(VALTYPE));
}

template <typename VALTYPE>
void Grid<VALTYPE>::Laplacian(const Grid<VALTYPE> & grid, const VALTYPE& fStepX, const VALTYPE& fStepY, const VALTYPE& fHalfStepX, const VALTYPE& fHalfStepY)
{
	if (grid.nRows != nRows || grid.nCols != nCols || nRows < 3 || grid.nRows < 3)
	{
		return;
	}
	
#ifdef std11
	VALTYPE * pPrevRowValue = grid.ptrData + 1;
	VALTYPE * pCurRowValue = grid.ptrData + nCols + 1;
	VALTYPE * pNextRowValue = grid.ptrData + 2 * nCols + 1;
	VALTYPE * pCurDst = ptrData + nCols + 1;
	for (int y = 1; y < nRows - 1; ++y)
	{
		for (int x = 1; x < nCols - 1; ++x)
		{
			const VALTYPE fCurVal = *pCurRowValue;
			*pCurDst = (1.0 / fHalfStepY) * ((fCurVal - *pPrevRowValue) / fStepY -
				(*pNextRowValue - fCurVal) / fStepY) +
				(1.0 / fHalfStepX) *((fCurVal - *(pCurRowValue - 1)) / fStepX -
				(*(pCurRowValue + 1) - fCurVal) / fStepX);
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
#else
	VALTYPE * pSrc = grid.ptrData;
	VALTYPE * pDst = ptrData;
#ifdef OMP
	#pragma omp parallel for collapse(2)
#endif
	for (int y = 1; y < nRows - 1; ++y)
	{
		for (int x = 1; x < nCols - 1; ++x)
		{
			const VALTYPE fCurVal = pSrc[y*nCols+x];
			pDst[y*nCols+x] = (1.0 / fHalfStepY) * ((fCurVal - pSrc[(y-1)*nCols+x]) / fStepY -
				(pSrc[(y + 1)*nCols + x] - fCurVal) / fStepY) +
				(1.0 / fHalfStepX) *((fCurVal - pSrc[y*nCols + x - 1]) / fStepX -
				(pSrc[y*nCols + x + 1] - fCurVal) / fStepX);
		}
	}
#endif
}

template <typename VALTYPE>
VALTYPE Grid<VALTYPE>::MaxNormDifference() const
{
	VALTYPE fMax = -std::numeric_limits<VALTYPE>::min();
#ifdef std11
	VALTYPE * pCurSrc = ptrData + nCols + 1;

	for (int i = 1; i < nRows - 1; ++i)
	{
		for (int j = 1; j < nCols - 1; ++j)
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
#else
#ifdef OMP
	#pragma omp parallel for collapse(2)
#endif
	VALTYPE * pSrc = ptrData;

	for (int i = 1; i < nRows - 1; ++i)
	{
		for (int j = 1; j < nCols - 1; ++j)
		{
			auto fDiff = fabs(pSrc[i*nCols+j]);
			if (fDiff > fMax)
			{
				fMax = fDiff;
			}
		}
	}
#endif
	return fMax;
}

#ifdef std11
template <typename VALTYPE>
void Grid<VALTYPE>::Fill(std::function<VALTYPE(VALTYPE, VALTYPE)> Func_, FillTypes fillType,
						const VALTYPE x0, const VALTYPE y0, const VALTYPE fStepX, const VALTYPE fStepY)
{
	VALTYPE * pValues = ptrData;
	if (fillType == FILL_BOUNDARY)
	{
		VALTYPE x1 = (nCols-1)*fStepX + x0;
		VALTYPE y1 = (nRows-1)*fStepY + y0;
		memset(pValues, VALTYPE(0), nRows * nCols * sizeof(VALTYPE));
		for (int y = 0; y < nRows; ++y)
		{
			pValues[y*nCols] = Func_(x0, y0+y*fStepY);
			pValues[(y + 1)*nCols - 1] = Func_(x1, y0 + y*fStepY);
		}
		for (int x = 0; x < nCols; ++x)
		{
			pValues[x] = Func_(x0+x*fStepX, y0);
			pValues[(nRows - 1)*nCols + x] = Func_(x0+x*fStepX, y1);
		}
	}
	else
	{
		auto * pCurValue = pValues;
		for (int y = 0; y < nRows; ++y)
		{
			for (int x = 0; x < nCols; ++x)
			{
				*pCurValue = Func_(x0+x*fStepX,y0+y*fStepY);
				pCurValue++;
			}
		}
	}
}
#else
template <typename VALTYPE>
void Grid<VALTYPE>::Fill(VALTYPE (*Func_)(VALTYPE, VALTYPE), FillTypes fillType,
	const VALTYPE x0, const VALTYPE y0, const VALTYPE fStepX, const VALTYPE fStepY)
{
	VALTYPE * pValues = ptrData;
	if (fillType == FILL_BOUNDARY)
	{
		VALTYPE x1 = (nCols - 1)*fStepX + x0;
		VALTYPE y1 = (nRows - 1)*fStepY + y0;
		memset(pValues, VALTYPE(0), nRows * nCols * sizeof(VALTYPE));
		for (int y = 0; y < nRows; ++y)
		{
			pValues[y*nCols] = Func_(x0, y0 + y*fStepY);
			pValues[(y + 1)*nCols - 1] = Func_(x1, y0 + y*fStepY);
		}
		for (int x = 0; x < nCols; ++x)
		{
			pValues[x] = Func_(x0 + x*fStepX, y0);
			pValues[(nRows - 1)*nCols + x] = Func_(x0 + x*fStepX, y1);
		}
	}
	else
	{
		VALTYPE * pCurValue = pValues;
		for (int y = 0; y < nRows; ++y)
		{
			for (int x = 0; x < nCols; ++x)
			{
				*pCurValue = Func_(x0 + x*fStepX, y0 + y*fStepY);
				pCurValue++;
			}
		}
	}
}
#endif

/*
template <typename VALTYPE>
GridStorage<VALTYPE>::GridStorage(const uint64_t nSize, const int nNumParts)
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
*/

template<typename VALTYPE>
VALTYPE DotProduct(const Grid<VALTYPE> & grid1, const Grid<VALTYPE> & grid2, const VALTYPE k)
{
	VALTYPE fResult = 0;
	const VALTYPE fPowK = k;
#ifdef std11
	VALTYPE * pCurMult1 = grid1.ptrData+grid1.nCols+1;
	VALTYPE * pCurMult2 = grid2.ptrData+grid2.nCols+1;
	for (int y = 1; y < grid1.nRows-1; ++y)
	{
		for (int x = 1; x < grid1.nCols-1; ++x)
		{
			fResult += fPowK * (*pCurMult1) * (*pCurMult2);
			++pCurMult1;
			++pCurMult2;
		}
		pCurMult1+=2;
		pCurMult2+=2;
	}
#else
	VALTYPE * pMult1 = grid1.ptrData;
	VALTYPE * pMult2 = grid2.ptrData;
#ifdef OMP
	#pragma omp parallel for
#endif
	for (int y = 1; y < grid1.nRows - 1; ++y)
	{
		for (int x = 1; x < grid1.nCols - 1; ++x)
		{
			fResult += fPowK * pMult1[y*grid1.nCols+x] * pMult2[y*grid1.nCols+x];
		}
	}
#endif
	return fResult;
}

template<typename VALTYPE>
VALTYPE DotProduct_MPI(const Grid<VALTYPE> & grid1, const Grid<VALTYPE> & grid2, const VALTYPE k, 
	const int numTopNeighb, const int numLeftNeighb)
{
	VALTYPE fResult = 0;
#ifdef std11
	VALTYPE * pCurMult1 = grid1.ptrData + (1+ numTopNeighb)*grid1.nCols + 1 + numLeftNeighb;
	VALTYPE * pCurMult2 = grid2.ptrData + (1 + numTopNeighb)*grid2.nCols + 1 + numLeftNeighb;
	for (int y = numTopNeighb; y < grid1.nRows - 1; ++y)
	{
		for (int x = numLeftNeighb; x < grid1.nCols - 1; ++x)
		{
			fResult += k * (*pCurMult1) * (*pCurMult2);
			++pCurMult1;
			++pCurMult2;
		}
		pCurMult1 += 2+numLeftNeighb;
		pCurMult2 += 2+numLeftNeighb;
	}
#else
	VALTYPE * pMult1 = grid1.ptrData;
	VALTYPE * pMult2 = grid2.ptrData;
#ifdef OMP
	#pragma omp parallel for collapse(2)
#endif
	for (int y = numTopNeighb; y < grid1.nRows - 1; ++y)
	{
		for (int x = numLeftNeighb; x < grid1.nCols - 1; ++x)
		{
			fResult += k * pMult1[y*grid1.nCols+x] * pMult2[y*grid1.nCols+x];
		}
	}
#endif
	return fResult;
}


template <typename VALTYPE>
inline VALTYPE EstimateTau(const Grid<VALTYPE>& g1, const Grid<VALTYPE>& g2,
	const Grid<VALTYPE>& g3, const Grid<VALTYPE>& g4, const VALTYPE k)
{
	return DotProduct(g1, g2, k) / DotProduct(g3, g4, k)
}


template <typename VALTYPE>
GridMPI<VALTYPE>::GridMPI(const int nRows_, const int nCols_, const int nProcIndex_, const int nNumProc)
{
	nRows = nRows_;
	nCols = nCols_;
	nProcIndex = nProcIndex_;
	ptrData = new VALTYPE[nRows_ * nCols_ * sizeof(VALTYPE)];
	memset(ptrData, VALTYPE(0), nRows_ * nCols_ * sizeof(VALTYPE));

	//estimate neighborhood
	int pow2 = log2(nNumProc);
	int nProcRow = 1 << (pow2 / 2);
	int nProcCol = 1 << (pow2 / 2 + pow2 % 2);
	int nProcX = nProcIndex%nProcCol;
	int nProcY = nProcIndex / nProcCol;
	neighbors[3] = nProcY != 0 ? nProcIndex - nProcCol : -1;
	neighbors[2] = nProcY != nProcRow - 1 ? nProcIndex + nProcCol : -1;
	neighbors[1] = nProcX != 0 ? nProcIndex - 1 : -1;
	neighbors[0] = nProcX != nProcCol - 1 ? nProcIndex + 1 : -1;
	//std::cout << nProcRow << ' ' << nProcCol << std::endl;
	//std::cout << nProcIndex <<' '<<nProcY<<' '<<nProcX<< ' ' << neighbors[3] << ' ' << neighbors[2] << ' ' << neighbors[1] << ' ' << neighbors[0] << std::endl;

	//allocate buffers to send-recieve
	pTopBuf = hasTopNeighb()?new VALTYPE[2 * nCols * sizeof(VALTYPE)]:0;
	pBotBuf = hasBotNeighb() ? new VALTYPE[2 * nCols * sizeof(VALTYPE)] : 0;
	pLeftBuf = hasLeftNeighb() ? new VALTYPE[2 * nRows * sizeof(VALTYPE)] : 0;
	pRightBuf = hasRightNeighb() ? new VALTYPE[2 * nRows * sizeof(VALTYPE)] : 0;

}

template <typename VALTYPE>
GridMPI<VALTYPE>::~GridMPI()
{
	if (pTopBuf)
	{
		delete[] pTopBuf;
	}
	if (pBotBuf)
	{
		delete[] pBotBuf;
	}
	if (pLeftBuf)
	{
		delete[] pLeftBuf;
	}
	if (pRightBuf)
	{
		delete[] pRightBuf;
	}
}

#ifdef std11
template <typename VALTYPE>
void GridMPI<VALTYPE>::Fill(std::function<VALTYPE(VALTYPE, VALTYPE)> Func_, FillTypes fillType,
	const VALTYPE x0, const VALTYPE y0, const VALTYPE fStepX, const VALTYPE fStepY)
{
	VALTYPE * pValues = ptrData;
	if (fillType == FILL_BOUNDARY)
	{
		VALTYPE x1 = (nCols - 1)*fStepX + x0;
		VALTYPE y1 = (nRows - 1)*fStepY + y0;
		memset(pValues, VALTYPE(0), nRows * nCols * sizeof(VALTYPE));
		if (!hasLeftNeighb())
		{
			for (int y = 0; y < nRows; ++y)
			{
				pValues[y*nCols] = Func_(x0, y0 + y*fStepY);

			}
		}
		if (!hasTopNeighb())
		{
			for (int x = 0; x < nCols; ++x)
			{
				pValues[x] = Func_(x0 + x*fStepX, y0);

			}
		}
		if (!hasRightNeighb())
		{
			for (int y = 0; y < nRows; ++y)
			{
				pValues[(y + 1)*nCols - 1] = Func_(x1, y0 + y*fStepY);
			}
		}
		if (!hasBotNeighb())
		{
			for (int x = 0; x < nCols; ++x)
			{
				pValues[(nRows - 1)*nCols + x] = Func_(x0 + x*fStepX, y1);
			}
		}
	}
	else
	{
		auto * pCurValue = pValues;
		for (int y = 0; y < nRows; ++y)
		{
			for (int x = 0; x < nCols; ++x)
			{
				*pCurValue = Func_(x0 + x*fStepX, y0 + y*fStepY);
				pCurValue++;
			}
		}
	}
}
#else
template <typename VALTYPE>
void GridMPI<VALTYPE>::Fill(VALTYPE(*Func_)(VALTYPE, VALTYPE), FillTypes fillType,
	const VALTYPE x0, const VALTYPE y0, const VALTYPE fStepX, const VALTYPE fStepY)
{
	VALTYPE * pValues = ptrData;
	if (fillType == FILL_BOUNDARY)
	{
		VALTYPE x1 = (nCols - 1)*fStepX + x0;
		VALTYPE y1 = (nRows - 1)*fStepY + y0;
		memset(pValues, VALTYPE(0), nRows * nCols * sizeof(VALTYPE));
		if (!hasLeftNeighb())
		{
			for (int y = 0; y < nRows; ++y)
			{
				pValues[y*nCols] = Func_(x0, y0 + y*fStepY);

			}
		}
		if (!hasTopNeighb())
		{
			for (int x = 0; x < nCols; ++x)
			{
				pValues[x] = Func_(x0 + x*fStepX, y0);

			}
		}
		if (!hasRightNeighb())
		{
			for (int y = 0; y < nRows; ++y)
			{
				pValues[(y + 1)*nCols - 1] = Func_(x1, y0 + y*fStepY);
			}
		}
		if (!hasBotNeighb())
		{
			for (int x = 0; x < nCols; ++x)
			{
				pValues[(nRows - 1)*nCols + x] = Func_(x0 + x*fStepX, y1);
			}
		}
	}
	else
	{
		VALTYPE * pCurValue = pValues;
		for (int y = 0; y < nRows; ++y)
		{
			for (int x = 0; x < nCols; ++x)
			{
				*pCurValue = Func_(x0 + x*fStepX, y0 + y*fStepY);
				pCurValue++;
			}
		}
	}
}
#endif

template <typename VALTYPE>
void GridMPI<VALTYPE>::operator = (const GridMPI<VALTYPE> & grid)
{
	//if (ptrData && (nRows * nCols) != grid.nRows * grid.nCols)
	//{
	//	delete[] ptrData;
	//	ptrData = 0;
	//}
	//else if ((nRows * nCols) != grid.nRows * grid.nCols)
	//{
	//	ptrData = new VALTYPE[grid.nCols * grid.nRows * sizeof(VALTYPE)];
	//}

	nRows = grid.nRows;
	nCols = grid.nCols;
	VALTYPE * pDst = ptrData;
	VALTYPE * pSrc = grid.ptrData;
	memcpy(pDst, pSrc, grid.nRows * grid.nCols * sizeof(VALTYPE));
}

template <typename VALTYPE>
void GridMPI<VALTYPE>::Send(const int tag, const MPI_Comm & comm)
{
	MPI_Request Req_mpi;
	if (hasTopNeighb())
	{
		VALTYPE * pDst = pTopBuf+nCols;
		VALTYPE * pSrc = ptrData;
		memcpy(pDst, pSrc + nCols, nCols * sizeof(VALTYPE));
		//std::cout << nProcIndex << " Try SEND to " << neighbors[3] << std::endl;
		MPI_Isend(pDst, nCols, MPI_FLOAT, neighbors[3], tag, comm, &Req_mpi);
	}
	if (hasBotNeighb())
	{
		VALTYPE * pDst = pBotBuf+nCols;
		VALTYPE * pSrc = ptrData;
		memcpy(pDst, pSrc + (nRows - 2)*nCols, nCols * sizeof(VALTYPE));
			//std::cout << nProcIndex << " Try SEND to " << neighbors[2] << std::endl;
		MPI_Isend(pDst, nCols, MPI_FLOAT, neighbors[2], tag, comm, &Req_mpi);
	}
	if (hasLeftNeighb())
	{
		VALTYPE * pDst = pLeftBuf+nRows;
		VALTYPE * pSrc = ptrData;
		for (int y = 0; y < nRows; ++y)
		{
			*pDst = pSrc[y*nCols + 1];
			++pDst;
		}
		//std::cout << nProcIndex << " Try SEND to " << neighbors[1] << std::endl;
		MPI_Isend(pLeftBuf+nRows, nRows, MPI_FLOAT, neighbors[1], tag, comm, &Req_mpi);
	}
	if (hasRightNeighb())
	{
		VALTYPE * pDst = pRightBuf+nRows;
		VALTYPE * pSrc = ptrData;
		for (int y = 0; y < nRows; ++y)
		{
			*pDst = pSrc[y*nCols + nCols - 2];
			++pDst;
		}
		//std::cout << nProcIndex << " Try SEND to " << neighbors[0] << std::endl;
		MPI_Isend(pRightBuf+nRows, nRows, MPI_FLOAT, neighbors[0], tag, comm, &Req_mpi);
	}
}

template <typename VALTYPE>
void GridMPI<VALTYPE>::Refresh(const int tag, const MPI_Comm & comm)
{
	MPI_Request Req_mpi[4];
	int nNumReq = 0;
	if (hasTopNeighb())
	{
		MPI_Irecv(pTopBuf, nCols, MPI_FLOAT, neighbors[3], tag, comm, &Req_mpi[nNumReq++]);
	}
	if (hasBotNeighb())
	{
		MPI_Irecv(pBotBuf, nCols, MPI_FLOAT, neighbors[2], tag, comm, &Req_mpi[nNumReq++]);
	}
	if (hasLeftNeighb())
	{
		MPI_Irecv(pLeftBuf, nRows, MPI_FLOAT, neighbors[1], tag, comm, &Req_mpi[nNumReq++]);
	}
	if (hasRightNeighb())
	{
		MPI_Irecv(pRightBuf, nRows, MPI_FLOAT, neighbors[0], tag, comm, &Req_mpi[nNumReq++]);
	}	
	if (nNumReq)
	{
		MPI_Status Stat_mpi[4];
		MPI_Waitall(nNumReq, Req_mpi, Stat_mpi);
	}

	if (hasTopNeighb())
	{
		VALTYPE * pSrc = pTopBuf;
		VALTYPE * pDst = ptrData;
		memcpy(pDst, pSrc, nCols * sizeof(VALTYPE));
	}
	if (hasBotNeighb())
	{
		VALTYPE * pSrc = pBotBuf;
		VALTYPE * pDst = ptrData;
		memcpy(pDst + (nRows - 1)*nCols, pSrc, nCols * sizeof(VALTYPE));
	}	
	if (hasLeftNeighb())
	{
		VALTYPE * pSrc = pLeftBuf;
		VALTYPE * pDst = ptrData;
		for (int y = 0; y < nRows; ++y)
		{
			pDst[y*nCols] = *pSrc;
			++pSrc;
		}
	}
	if (hasRightNeighb())
	{
		VALTYPE * pSrc = pRightBuf;
		VALTYPE * pDst = ptrData;
		for (int y = 0; y < nRows; ++y)
		{
			pDst[y*nCols + nCols - 1] = *pSrc;
			++pSrc;
		}
	}
}

template <typename VALTYPE>
inline VALTYPE EstimateTauMPI(const GridMPI<VALTYPE>& g1, const GridMPI<VALTYPE>& g2,
								const GridMPI<VALTYPE>& g3, const GridMPI<VALTYPE>& g4,
								const VALTYPE k, const MPI_Comm& comm)
{
	VALTYPE tau = 0;
	VALTYPE GlobalTau = 0;
	int topNeighb = g1.hasTopNeighb() ? 1 : 0;
	int leftNeighb = g1.hasLeftNeighb() ? 1 : 0;
	VALTYPE LocalTau = DotProduct_MPI(g1, g2, k, topNeighb, leftNeighb);
	//std::cout <<k<<' '<< LocalTau << ' ';
	MPI_Allreduce(&LocalTau, &GlobalTau, 1, MPI_FLOAT, MPI_SUM, comm);
	tau = GlobalTau;
	LocalTau = DotProduct_MPI(g3, g4, k, topNeighb, leftNeighb);
	//std::cout << LocalTau << ' ';
	MPI_Allreduce(&LocalTau, &GlobalTau, 1, MPI_FLOAT, MPI_SUM, comm);
	//std::cout << tau << ' ' << GlobalTau << ' ' << tau / GlobalTau << std::endl;
	return tau = GlobalTau ? tau / GlobalTau : 0;
}

template <typename VALTYPE>
inline VALTYPE EstimateGlobalDiffMPI(const GridMPI<VALTYPE>& grid, const MPI_Comm& comm)
{
	float GlobalDiff = 0;
	float LocalDiff = grid.MaxNormDifference();
	MPI_Allreduce(&LocalDiff, &GlobalDiff, 1, MPI_FLOAT, MPI_MAX, comm);
	return GlobalDiff;
}

void Wait(MPI_Comm comm)
{
	int nCurIndex = 0;
	int nSize = 0;
	MPI_Comm_size(comm, &nSize);
	MPI_Comm_rank(comm, &nCurIndex);
	if (nSize == 1)
		return;
	int nPrevProcIndex = (nCurIndex - 1)<0?nSize-1: nCurIndex - 1;
	int buf = 0;
	MPI_Request req;
	MPI_Irecv(&buf, 1, MPI_INT, nPrevProcIndex, 0, comm, &req);
	MPI_Status status;
	MPI_Wait(&req, &status);
}

void Sig(MPI_Comm comm)
{
	int nCurIndex = 0;
	int nSize = 0;
	MPI_Comm_size(comm, &nSize);
	MPI_Comm_rank(comm, &nCurIndex);
	if (nSize == 1)
		return;
	int nNextProcIndex = (nCurIndex+1)%nSize;

	int buf = 0;
	MPI_Request status;
	MPI_Isend(&buf, 1, MPI_INT, nNextProcIndex, 0, comm, &status);
}