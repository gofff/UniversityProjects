
template <typename VALTYPE>
DirichletProblem<VALTYPE>::DirichletProblem(const VALTYPE x0_, const VALTYPE y0_,
								const VALTYPE x1_, const VALTYPE y1_,
								const uint32_t nPointsX_, const uint32_t nPointsY_,
								const std::function<VALTYPE(const VALTYPE, const VALTYPE)> BoundaryFunc_,
								const std::function<VALTYPE(const VALTYPE, const VALTYPE)> AppFunc_)
	: x0(x0_)
	, x1(x1_)
	, y0(y0_)
	, y1(y1_)
	, nPointsY(nPointsY_)
	, nPointsX(nPointsX_)
	, eps(1e-6)
{
	fStepX = (x1 - x0) / (nPointsX - 1);
	fStepY = (y1 - y0) / (nPointsY - 1);
	fHalfStepX = fStepX;
	BoundaryFunction = BoundaryFunc_;
	AppFunction = AppFunc_;
}


//#define IT_LOG
template <typename VALTYPE>
VALTYPE DirichletProblem<VALTYPE>::Solve()
{
	//GridStorage<VALTYPE> storage(nPointsY*nPointsX, 7);
	Grid<VALTYPE> p(nPointsY, nPointsX);
	p.Fill(BoundaryFunction, FILL_BOUNDARY,x0,y0,fStepX,fStepY);
	p.dump("P");
	Grid<VALTYPE> F(nPointsY, nPointsX);
	F.Fill(AppFunction, FILL_ALL, x0, y0, fStepX, fStepY);
	F.dump("F");
	Grid<VALTYPE> r(nPointsY, nPointsX);
	Grid<VALTYPE> g(nPointsY, nPointsX);
	Grid<VALTYPE> g_1(nPointsY, nPointsX);
	Grid<VALTYPE> rLap(nPointsY, nPointsX);
	Grid<VALTYPE> gLap(nPointsY, nPointsX);
	VALTYPE tau = 0;
	VALTYPE alpha = 0;
	VALTYPE diff = 0;
	VALTYPE k = fStepX*fStepY; //to dot product

	//first iteration
	r.Laplacian(p,fStepX, fStepY, fHalfStepX);
	r.dump("Lap(p)");
	r -= F; //r_{1} = d(p) - F
	r.dump("Lap(p)-F");
	g_1 = r; //запомнили для следующей итерации g_{1}
	rLap.Laplacian(r, fStepX, fStepY, fHalfStepX);// d(r_{1})
	rLap.dump("Lap(Lap(p)-F)");
	tau = DotProduct(r, r, k) / DotProduct(rLap, r, k); 
	std::cout <<"tau= "<< DotProduct(r, r, k)<<' '<< DotProduct(rLap, r, k)<<' '<< tau << std::endl;
	r *= tau; 
	r.dump("r*tau");
	diff = r.MaxNormDifference(); // |p_{1} - p_{0}| = tau*r
	p -= r;
	p.dump("p-r");

	num_iter = 1;
	//other iterations
	while (diff > eps)
	{
		//std::cout << "Iteration: " << num_iter << std::endl;
		++num_iter;
		r.Laplacian(p, fStepX,fStepY, fHalfStepX);
		r.dump("Lap(r)");
		r -= F; // r_{k}
		r.dump("-F");
		rLap.Laplacian(r, fStepX, fStepY, fHalfStepX);
		rLap.dump("Lap(Lap(r))");
		gLap.Laplacian(g_1, fStepX, fStepY, fHalfStepX);
		gLap.dump("gLap");
		if (memcmp(rLap.ptrData.get(), gLap.ptrData.get(), gLap.nCols*gLap.nRows * sizeof(VALTYPE)))
		{
			alpha = DotProduct(rLap, g_1, k) / DotProduct(gLap, g_1, k);
			std::cout <<"a= "<< alpha << std::endl;
		}
		else
		{
			alpha = 0.0;
		}
		
		g_1 *= alpha; 
		g_1.dump("g_{k-1}*alpha");
		g = r;
		g -= g_1; //g_{k}=r_{k}-alpha*g_{k-1}
		g.dump("g_{k}=r_{k}-alpha*g_{k-1}");
		g_1 = g; //запомнили
		gLap.Laplacian(g, fStepX, fStepY, fHalfStepX);
		gLap.dump("Lap(g)");
		if (memcmp(gLap.ptrData.get(), g.ptrData.get(), g.nCols*g.nRows * sizeof(VALTYPE)))
		{
			tau = DotProduct(r, g, k) / DotProduct(gLap, g, k);
			std::cout << "tau= " << tau << std::endl;
		}
		else
		{
			tau = 0.0;
		}
		g *= tau; 
		g.dump("g*tau");

		diff = g.MaxNormDifference();
		p -= g; // p_{k+1} = p_{k} - alpha*g_{k}
		p.dump("p");
		/////////////////////////////////
		float maxV = -std::numeric_limits<VALTYPE>::min();
		VALTYPE * data = p.ptrData.get();
		for (int i = 1; i < p.nRows - 1; ++i)
		{
			for (int j = 1; j < p.nCols - 1; ++j)
			{
				double diffV = fabs(data[i*p.nCols + j] - BoundaryFunction(x0 + j*fStepX, y0 + i*fStepX));
				if (diffV > maxV)
				{
					maxV = diffV;
				}
			}
		}
		////////////////////////////////////
		std::cout << num_iter <<' '<<diff<<' '<<maxV<< std::endl;
		
		
	}
	solution=p;
	error = diff;
	getchar();
	return 0;
}

template <typename VALTYPE>
void DirichletProblem<VALTYPE>::InitSubproblem_mpi()
{
	// Get the number of processes
	int nNumOfProc;
	MPI_Comm_size(Comm_mpi, &nNumOfProc);

	// Get the rank of the process
	MPI_Comm_rank(Comm_mpi, &nProcIndex);
	
	RememberGlobalProblemParameters();

	int pow2 = log2(nNumOfProc);
	int nProcRow = 1<<(pow2/2);
	int nProcCol = 1<<(pow2/2+pow2%2);
	nPointsX = std::floor(VALTYPE(nPointsX) / nProcCol + 0.5);
	nPointsY = std::floor(VALTYPE(nPointsY) / nProcRow + 0.5);
	int nProcX = nProcIndex%nProcCol;
	int nProcY = nProcIndex/nProcCol;
	VALTYPE segLenX = (x1 - x0) / nProcCol;
	VALTYPE segLenY = VALTYPE(y1 - y0) / nProcRow;
	x0 = segLenX*nProcX;
	x1 = x0 + segLenX;
	y0 = segLenY*nProcY;
	y1 = y0 + segLenY;
	fStepX = (x1 - x0) / (nPointsX-1);
	fStepY = (y1 - y0) / (nPointsY-1);
	
	
	Neighbors.TopNeighb = nProcY != 0 ? nProcIndex-nProcCol : -1;
	Neighbors.BotNeighb = nProcY != nProcRow-1 ? nProcIndex + nProcCol : -1;
	Neighbors.LeftNeighb = nProcX != 0 ? nProcIndex - 1 : -1;
	Neighbors.RightNeighb = nProcX != nProcCol - 1 ? nProcIndex + 1 : -1;
	//std::cout << nProcIndex <<" as "<<nProcY<<' '<<nProcX<<" Has Neighbours " << Neighbors.TopNeighb << ' '
	//	<< Neighbors.BotNeighb << ' ' << Neighbors.LeftNeighb << ' ' << Neighbors.RightNeighb << std::endl;

	y0 -= Neighbors.TopNeighb != -1 ? fStepY : 0;
	nPointsY += Neighbors.TopNeighb != -1 ? 1 : 0;

	y1 -= Neighbors.BotNeighb != -1 ? fStepY : 0;
	nPointsY += Neighbors.BotNeighb != -1 ? 1 : 0;

	x0 -= Neighbors.LeftNeighb != -1 ? fStepY : 0;
	nPointsX += Neighbors.LeftNeighb != -1 ? 1 : 0;

	x1 -= Neighbors.RightNeighb != -1 ? fStepY : 0;
	nPointsX += Neighbors.RightNeighb != -1 ? 1 : 0;

	BBufer.nSizeX = nPointsX;
	BBufer.nSizeY = nPointsY;
	BBufer.ptrTop = std::shared_ptr<VALTYPE>(new VALTYPE[BBufer.nSizeX * sizeof(VALTYPE)]);
	memset(BBufer.ptrTop.get(), VALTYPE(0), BBufer.nSizeX * sizeof(VALTYPE));
	BBufer.ptrBot = std::shared_ptr<VALTYPE>(new VALTYPE[BBufer.nSizeX * sizeof(VALTYPE)]);
	memset(BBufer.ptrBot.get(), VALTYPE(0), BBufer.nSizeX * sizeof(VALTYPE));
	BBufer.ptrLeft = std::shared_ptr<VALTYPE>(new VALTYPE[BBufer.nSizeY * sizeof(VALTYPE)]);
	memset(BBufer.ptrLeft.get(), VALTYPE(0), BBufer.nSizeY * sizeof(VALTYPE));
	BBufer.ptrRight = std::shared_ptr<VALTYPE>(new VALTYPE[BBufer.nSizeY * sizeof(VALTYPE)]);
	memset(BBufer.ptrRight.get(), VALTYPE(0), BBufer.nSizeY * sizeof(VALTYPE));

	SendBufer.nSizeX = nPointsX;
	SendBufer.nSizeY = nPointsY;
	SendBufer.ptrTop = std::shared_ptr<VALTYPE>(new VALTYPE[SendBufer.nSizeX * sizeof(VALTYPE)]);
	memset(SendBufer.ptrTop.get(), VALTYPE(0), SendBufer.nSizeX * sizeof(VALTYPE));
	SendBufer.ptrBot = std::shared_ptr<VALTYPE>(new VALTYPE[SendBufer.nSizeX * sizeof(VALTYPE)]);
	memset(SendBufer.ptrBot.get(), VALTYPE(0), SendBufer.nSizeX * sizeof(VALTYPE));
	SendBufer.ptrLeft = std::shared_ptr<VALTYPE>(new VALTYPE[SendBufer.nSizeY * sizeof(VALTYPE)]);
	memset(SendBufer.ptrLeft.get(), VALTYPE(0), SendBufer.nSizeY * sizeof(VALTYPE));
	SendBufer.ptrRight = std::shared_ptr<VALTYPE>(new VALTYPE[SendBufer.nSizeY * sizeof(VALTYPE)]);
	memset(SendBufer.ptrRight.get(), VALTYPE(0), SendBufer.nSizeY * sizeof(VALTYPE));

	if (Neighbors.TopNeighb == -1)
	{
		VALTYPE * ptrData = BBufer.ptrTop.get();
		for (int x = 0; x < BBufer.nSizeX; ++x)
		{
			*ptrData = BoundaryFunction(x0 + x*fStepX, y0);
			++ptrData;
		}
	}
	if (Neighbors.BotNeighb == -1)
	{
		VALTYPE * ptrData = BBufer.ptrBot.get();
		for (int x = 0; x < BBufer.nSizeX; ++x)
		{
			*ptrData = BoundaryFunction(x0 + x*fStepX, y1);
			++ptrData;
		}
	}
	if (Neighbors.LeftNeighb == -1)
	{
		VALTYPE * ptrData = BBufer.ptrLeft.get();
		for (int y = 0; y < BBufer.nSizeY; ++y)
		{
			*ptrData = BoundaryFunction(x0, y0+y*fStepY);
			++ptrData;
		}
	}

	if (Neighbors.RightNeighb == -1)
	{
		VALTYPE * ptrData = BBufer.ptrRight.get();
		for (int y = 0; y < BBufer.nSizeY; ++y)
		{
			*ptrData = BoundaryFunction(x1, y0 + y*fStepY);
			++ptrData;
		}
	}
	fAvStepX = fStepX;
	fAvStepY = fStepY;
}

template <typename VALTYPE>
void DirichletProblem<VALTYPE>::RememberGlobalProblemParameters()
{
	nPointsX_nompi = nPointsX;
	nPointsY_nompi = nPointsY;
	x0_nompi = x0;
	x1_nompi = x1;
	y0_nompi = y0;
	y1_nompi = y1;
	fStepX_nompi = fStepX;
	fStepY_nompi = fStepY;
	fAvStepX_nompi = fAvStepX;
	fAvStepY_nompi = fAvStepY;
}

template <typename VALTYPE>
void DirichletProblem<VALTYPE>::RefreshBoundaries(Grid<VALTYPE> & grid, const uint32_t nIterNum)
{
	if (nIterNum)
	{
		MPI_Request Req_mpi[4];
		int nNumReq = 0;
		if (Neighbors.TopNeighb != -1)
		{
		//	std::cout << nProcIndex << " Try RECV from " << Neighbors.TopNeighb << std::endl;
			MPI_Irecv(BBufer.ptrTop.get(), nPointsX, MPI_FLOAT, Neighbors.TopNeighb, nIterNum, Comm_mpi, &Req_mpi[nNumReq++]);
		}
		if (Neighbors.BotNeighb != -1)
		{
		//	std::cout << nProcIndex << " Try RECV from " << Neighbors.BotNeighb << std::endl;
			MPI_Irecv(BBufer.ptrBot.get(), nPointsX, MPI_FLOAT, Neighbors.BotNeighb, nIterNum, Comm_mpi, &Req_mpi[nNumReq++]);
		}
		if (Neighbors.LeftNeighb != -1)
		{
		//	std::cout << nProcIndex << " Try RECV from " << Neighbors.LeftNeighb << std::endl;
			MPI_Irecv((void *)BBufer.ptrLeft.get(), nPointsY, MPI_FLOAT, Neighbors.LeftNeighb, nIterNum, Comm_mpi, &Req_mpi[nNumReq++]);
		}
		if (Neighbors.RightNeighb != -1)
		{
		//	std::cout << nProcIndex << " Try RECV from " << Neighbors.RightNeighb << std::endl;
			MPI_Irecv((void *)BBufer.ptrRight.get(), nPointsY, MPI_FLOAT, Neighbors.RightNeighb, nIterNum, Comm_mpi, &Req_mpi[nNumReq++]);
		}
		if (nNumReq)
		{
			MPI_Status Stat_mpi[4];
			//std::cout << nProcIndex << " Try WAITALL " <<nIterNum<< std::endl;
			MPI_Waitall(nNumReq, Req_mpi, Stat_mpi);
		}
		//std::cout << nProcIndex << " Success WAITALL " << nIterNum<< std::endl;
	}
	if (Neighbors.TopNeighb != -1 || !nIterNum)
	{
		VALTYPE * pSrc = BBufer.ptrTop.get();
		VALTYPE * pDst = grid.ptrData.get();
		memcpy(pDst, pSrc, BBufer.nSizeX * sizeof(VALTYPE));
	}
	if (Neighbors.BotNeighb != -1 || !nIterNum)
	{
		VALTYPE * pSrc = BBufer.ptrBot.get();
		VALTYPE * pDst = grid.ptrData.get();
		memcpy(pDst+(grid.nRows-1)*grid.nCols, pSrc, BBufer.nSizeX * sizeof(VALTYPE));
	}
	if (Neighbors.LeftNeighb != -1 || !nIterNum)
	{
		VALTYPE * pSrc = BBufer.ptrLeft.get();
		VALTYPE * pDst = grid.ptrData.get();
		for (int y = 0; y < BBufer.nSizeY; ++y)
		{
			pDst[(y)*grid.nCols] = *pSrc;
			++pSrc;
		}
	}
	if (Neighbors.RightNeighb != -1 || !nIterNum)
	{
		VALTYPE * pSrc = BBufer.ptrRight.get();
		VALTYPE * pDst = grid.ptrData.get();
		for (int y = 0; y < BBufer.nSizeY; ++y)
		{
			pDst[(y)*grid.nCols+grid.nCols-1] = *pSrc;
			++pSrc;
		}
	}
}

template <typename VALTYPE>
void DirichletProblem<VALTYPE>::SendBoundaries(const Grid<VALTYPE> & grid, const uint32_t nIterNum)
{
	MPI_Request Req_mpi;
	if (Neighbors.TopNeighb != -1)
	{
		VALTYPE * pDst = SendBufer.ptrTop.get();
		VALTYPE * pSrc = grid.ptrData.get();
		memcpy(pDst, pSrc+2*grid.nCols, SendBufer.nSizeX * sizeof(VALTYPE));
		//std::cout << nProcIndex << " Try SEND to " << Neighbors.TopNeighb << std::endl;
		MPI_Isend(pDst, SendBufer.nSizeX, MPI_FLOAT, Neighbors.TopNeighb, nIterNum, Comm_mpi, &Req_mpi);
	}
	if (Neighbors.BotNeighb != -1)
	{
		VALTYPE * pDst = SendBufer.ptrBot.get();
		VALTYPE * pSrc = grid.ptrData.get();
		memcpy(pDst, pSrc+ (grid.nRows - 3)*grid.nCols, SendBufer.nSizeX * sizeof(VALTYPE));
	//	std::cout << nProcIndex << " Try SEND to " << Neighbors.BotNeighb << std::endl;
		MPI_Isend(pDst, SendBufer.nSizeX, MPI_FLOAT, Neighbors.BotNeighb, nIterNum, Comm_mpi, &Req_mpi);
	}
	if (Neighbors.LeftNeighb != -1)
	{
		VALTYPE * pDst = SendBufer.ptrLeft.get();
		VALTYPE * pSrc = grid.ptrData.get();
		for (int y = 0; y < SendBufer.nSizeY; ++y)
		{
			*pDst = pSrc[y*grid.nCols+2];
			++pDst;
		}
//		std::cout << nProcIndex << " Try SEND to " << Neighbors.LeftNeighb << std::endl;
		MPI_Isend(SendBufer.ptrLeft.get(), SendBufer.nSizeY, MPI_FLOAT, Neighbors.LeftNeighb, nIterNum, Comm_mpi, &Req_mpi);
	}
	if (Neighbors.RightNeighb != -1)
	{
		VALTYPE * pDst = SendBufer.ptrRight.get();
		VALTYPE * pSrc = grid.ptrData.get();
		for (int y = 0; y < SendBufer.nSizeY; ++y)
		{
			*pDst = pSrc[y*grid.nCols + grid.nCols - 3];
			++pDst;
		}
	//	std::cout << nProcIndex << " Try SEND to " << Neighbors.RightNeighb << std::endl;
		MPI_Isend(SendBufer.ptrRight.get(), SendBufer.nSizeY, MPI_FLOAT, Neighbors.RightNeighb, nIterNum, Comm_mpi, &Req_mpi);
	}
}

template <typename VALTYPE>
void DirichletProblem<VALTYPE>::AgregateSolution()
{
	Grid<VALTYPE> generalSolution(nPointsX_nompi, nPointsY_nompi);
	MPI_Request Req_mpi;
	if (nProcIndex)
	{
		//MPI_Isend(solution.ptrData.get(), solution.nCols*solution.nRows, MPI_FLOAT, 0, 0, Comm_mpi, &Req_mpi);
	}
	else
	{

	}
}

template <typename VALTYPE>
VALTYPE DirichletProblem<VALTYPE>::SolveMPI()
{
	MPI_Init(NULL, NULL);
	Comm_mpi = MPI_COMM_WORLD;
	InitSubproblem_mpi();
	//while (1){
		Grid<VALTYPE> p(nPointsY, nPointsX);
		RefreshBoundaries(p, 0);
//		p.Fill(BoundaryFunction, FILL_BOUNDARY, x0, y0, fStepX, fStepY);
		Grid<VALTYPE> F(nPointsY, nPointsX);
		F.Fill(AppFunction, FILL_ALL, x0, y0, fStepX, fStepY);
		Grid<VALTYPE> r(nPointsY, nPointsX);
		Grid<VALTYPE> g(nPointsY, nPointsX);
		Grid<VALTYPE> g_1(nPointsY, nPointsX);
		Grid<VALTYPE> rLap(nPointsY, nPointsX);
		Grid<VALTYPE> gLap(nPointsY, nPointsX);
		VALTYPE tau = 0;
		VALTYPE alpha = 0;
		VALTYPE diff = 0;
		VALTYPE k = fHalfStepX; //to dot product
		int bHasLeft = Neighbors.LeftNeighb > -1 ? 1 : 0;
		int bHasTop = Neighbors.TopNeighb > -1 ? 1 : 0;


								//first iteration
		r.Laplacian(p, fStepX, fStepY, fHalfStepX);
		//r.dump("Lap(p)");
		r -= F; //r_{1} = d(p) - F
	//	if (!nProcIndex)
	//	{
	//		r.dump("R-F");
	//		Sig(Comm_mpi);
	//		Wait(Comm_mpi);
	//	}
	//	else
	//	{
	//		Wait(Comm_mpi);
	//		r.dump("R-F");
	//		Sig(Comm_mpi);
	//	}
		//r.dump("Lap(p)-F");
		g_1 = r; //запомнили для следующей итерации g_{1}
		MPI_Barrier(Comm_mpi);
		SendBoundaries(r, 1);
		RefreshBoundaries(r, 1);
		rLap.Laplacian(r, fStepX, fStepY, fHalfStepX);// d(r_{1})
	//	if (!nProcIndex)
	//	{
	//		rLap.dump("RLap");
	//		Sig(Comm_mpi);
	//		Wait(Comm_mpi);
	//	}
	//	else
	//	{
	//		Wait(Comm_mpi);
	//		rLap.dump("RLap");
	//		Sig(Comm_mpi);
	//	}
		//rLap.dump("Lap(Lap(p)-F)");
		//tau = DotProduct(r, r, k) / DotProduct(rLap, r, k);
		VALTYPE GlobalTau = 0;
		//VALTYPE LocalTau = DotProduct(r, r, k);
		VALTYPE LocalTau = DotProduct_MPI(r, r, k, bHasTop,bHasLeft);
		float chisl = LocalTau;
		MPI_Allreduce(&LocalTau, &GlobalTau, 1, MPI_FLOAT, MPI_SUM, Comm_mpi);
		tau = GlobalTau;
		//LocalTau = DotProduct(rLap, r, k);
		LocalTau = DotProduct_MPI(rLap, r, k, bHasTop, bHasLeft);
		float znamen = LocalTau;
		MPI_Allreduce(&LocalTau, &GlobalTau, 1, MPI_FLOAT, MPI_SUM, Comm_mpi);
		
		tau = GlobalTau ? tau / GlobalTau : 0;
		//std::cout << nProcIndex << " HAS TAU: " <<chisl<<' '<<znamen<<' '<< tau << std::endl;
		//std::cout <<"tau= "<< tau << std::endl;
		r *= tau;
		//r.dump("r*tau");
		//diff = r.MaxNormDifference(); // |p_{1} - p_{0}| = tau*r
		float GlobalDiff = 0;
		float LocalDiff = r.MaxNormDifference();
		MPI_Allreduce(&LocalDiff, &GlobalDiff, 1, MPI_FLOAT, MPI_MAX, Comm_mpi);
		diff = GlobalDiff;
		p -= r;
		
//		if (!nProcIndex)
//		{
//			p.dump("PBeforeSend");
//			Sig(Comm_mpi);
//			Wait(Comm_mpi);
//		}
//		else
//		{
//			Wait(Comm_mpi);
//			p.dump("PBeforeSend");
//			Sig(Comm_mpi);
//		}
		
		num_iter = 1;
		MPI_Barrier(Comm_mpi);
		SendBoundaries(p, 1);
		RefreshBoundaries(p, 1);
		//other iterations
		while (diff > eps)
		{
			//std::cout << "Iteration: " << num_iter << std::endl;
			++num_iter;
			
		//	if (!nProcIndex)
		//	{
		//		p.dump("PAfterSend");
		//		Sig(Comm_mpi);
		//		Wait(Comm_mpi);
		//	}
		//	else
		//	{
		//		Wait(Comm_mpi);
		//		p.dump("PAfterSend");
		//		Sig(Comm_mpi);
		//	}
			
			r.Laplacian(p, fStepX, fStepY, fHalfStepX);
			//r.dump("Lap(r)");
			r -= F; // r_{k}
			MPI_Barrier(Comm_mpi);
			SendBoundaries(r, num_iter+1);
			RefreshBoundaries(r, num_iter+1);
	//		if (!nProcIndex)
	//		{
	//			r.dump("r-=F");
	//			Sig(Comm_mpi);
	//			Wait(Comm_mpi);
	//		}
	//		else
	//		{
	//			Wait(Comm_mpi);
	//			r.dump("r-=F");
	//			Sig(Comm_mpi);
	//		}
			//r.dump("-F");
			rLap.Laplacian(r, fStepX, fStepY, fHalfStepX);
		//	if (!nProcIndex)
		//	{
		//		rLap.dump("rLap");
		//		Sig(Comm_mpi);
		//		Wait(Comm_mpi);
		//	}
		//	else
		//	{
		//		Wait(Comm_mpi);
		//		rLap.dump("rLap");
		//		Sig(Comm_mpi);
		//	}
			//rLap.dump("Lap(Lap(r))");
			SendBoundaries(g_1, num_iter);
			RefreshBoundaries(g_1, num_iter);
			gLap.Laplacian(g_1, fStepX, fStepY, fHalfStepX);
	//		if (!nProcIndex)
	//		{
	//			gLap.dump("gLap");
	//			Sig(Comm_mpi);
	//			Wait(Comm_mpi);
	//		}
	//		else
	//		{
	//			Wait(Comm_mpi);
	//			gLap.dump("gLap");
	//			Sig(Comm_mpi);
	//		}
			//gLap.dump("gLap");
			//if (memcmp(rLap.ptrData.get(), gLap.ptrData.get(), gLap.nCols*gLap.nRows * sizeof(VALTYPE)))
			//{
			//	alpha = DotProduct(rLap, g_1, k) / DotProduct(gLap, g_1, k);
			//	//std::cout <<"a= "<< alpha << std::endl;
			//}
			//else
			//{
			//	alpha = 0.0;
			//}
			VALTYPE GlobalAlpha = 0;
			//VALTYPE LocalAlpha = DotProduct(rLap, g_1, k);
			VALTYPE LocalAlpha = DotProduct_MPI(rLap, g_1, k, bHasTop, bHasLeft);
			MPI_Allreduce(&LocalAlpha, &GlobalAlpha, 1, MPI_FLOAT, MPI_SUM, Comm_mpi);
			alpha = GlobalAlpha;
			//LocalAlpha = DotProduct(gLap, g_1, k);
			LocalAlpha = DotProduct_MPI(gLap, g_1, k, bHasTop, bHasLeft);
			MPI_Allreduce(&LocalAlpha, &GlobalAlpha, 1, MPI_FLOAT, MPI_SUM, Comm_mpi);
			alpha = GlobalAlpha ? alpha / GlobalAlpha : 0;
			//std::cout << nProcIndex << " HAS ALPHA: " << alpha << std::endl;

			g_1 *= alpha;
		//	g_1.dump("g_{k-1}*alpha");
			g = r;
			g -= g_1; //g_{k}=r_{k}-alpha*g_{k-1}
		//	g.dump("g_{k}=r_{k}-alpha*g_{k-1}");
			g_1 = g; //запомнили
			SendBoundaries(g, num_iter);
			RefreshBoundaries(g, num_iter);
			gLap.Laplacian(g, fStepX, fStepY, fHalfStepX);
		//	gLap.dump("Lap(g)");
			//if (memcmp(gLap.ptrData.get(), g.ptrData.get(), g.nCols*g.nRows * sizeof(VALTYPE)))
			//{
			//	tau = DotProduct(r, g, k) / DotProduct(gLap, g, k);
			//	//std::cout << "tau= " << tau << std::endl;
			//}
			//else
			//{
			//	tau = 0.0;
			//}
			GlobalTau = 0;
			//LocalTau = DotProduct(r, g, k);
			LocalTau = DotProduct_MPI(r, g, k, bHasTop, bHasLeft);
			MPI_Allreduce(&LocalTau, &GlobalTau, 1, MPI_FLOAT, MPI_SUM, Comm_mpi);
			tau = GlobalTau;
			//LocalTau = DotProduct(gLap, g, k);
			LocalTau = DotProduct_MPI(gLap, g, k, bHasTop, bHasLeft);
			MPI_Allreduce(&LocalTau, &GlobalTau, 1, MPI_FLOAT, MPI_SUM, Comm_mpi);
			tau = GlobalTau ? tau / GlobalTau : 0;
			//std::cout << nProcIndex << " HAS TAU: " << chisl << ' ' << znamen << ' ' << tau << std::endl;
			g *= tau;
		//	g.dump("g*tau");
			//diff = g.MaxNormDifference();
			float GlobalDiff = 0;
			float LocalDiff = g.MaxNormDifference();
			MPI_Allreduce(&LocalDiff, &GlobalDiff, 1, MPI_FLOAT, MPI_MAX, Comm_mpi);
			diff = GlobalDiff;
			p -= g; // p_{k+1} = p_{k} - alpha*g_{k}
			
	//		if (!nProcIndex)
	//		{
	//			p.dump("PBefore");
	//			Sig(Comm_mpi);
	//			Wait(Comm_mpi);
	//		}
	//		else
	//		{
	//			Wait(Comm_mpi);
	//			p.dump("PBefore");
	//			Sig(Comm_mpi);
	//		}
			
			///////////////////////////////////////
			float maxV = -std::numeric_limits<VALTYPE>::min();
			VALTYPE * data = p.ptrData.get();
			for (int i = 1; i < p.nRows - 1; ++i)
			{
				for (int j = 1; j < p.nCols - 1; ++j)
				{
					double diffV = fabs(data[i*p.nCols + j] - BoundaryFunction(x0 + j*fStepX, y0 + i*fStepX));
					if (diffV > maxV)
					{
						maxV = diffV;
					}
				}
			}
			float GlobalErr = 0;
			float LocalErr = maxV;
			MPI_Allreduce(&LocalErr, &GlobalErr, 1, MPI_FLOAT, MPI_MAX, Comm_mpi);
			maxV = GlobalErr;
			//////////////////////////////////
	//		if (!nProcIndex)
	//		{
	//			std::cout << nProcIndex << ' ' << num_iter << ' ' << diff << ' ' << maxV << std::endl;
	//			Sig(Comm_mpi);
	//			Wait(Comm_mpi);
	//		}
	//		else
	//		{
	//			Wait(Comm_mpi);
	//			std::cout << nProcIndex << ' ' << num_iter << ' ' << diff << ' ' << maxV << std::endl;
	//			Sig(Comm_mpi);
	//		}
			std::cout << nProcIndex << ' ' << num_iter << ' ' << diff << ' '<<maxV<< std::endl;
			SendBoundaries(p, num_iter);
			RefreshBoundaries(p, num_iter);

	//		if (!nProcIndex)
	//		{
	//			p.dump("PAfter");
	//			Sig(Comm_mpi);
	//			Wait(Comm_mpi);
	//		}
	//		else
	//		{
	//			Wait(Comm_mpi);
	//			p.dump("PAfter");
	//			Sig(Comm_mpi);
	//		}
	//		Sig(Comm_mpi);
			//if (num_iter == 3)
			//	break;
		}
		solution = p;
		error = diff;

		std::cout << nProcIndex << " END" << std::endl;
		//Iteration2()
		//SendBoundaries();
		//RefreshBoundaries();
		AgregateSolution();

//	}
	MPI_Finalize();
	return 0;
}

