#ifdef std11
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
	fHalfStepY = fStepY;
	BoundaryFunction = BoundaryFunc_;
	AppFunction = AppFunc_;
}
#else
template <typename VALTYPE>
DirichletProblem<VALTYPE>::DirichletProblem(const VALTYPE x0_, const VALTYPE y0_,
	const VALTYPE x1_, const VALTYPE y1_,
	const uint32_t nPointsX_, const uint32_t nPointsY_,
	VALTYPE(*BoundaryFunc_)(const VALTYPE, const VALTYPE),
	VALTYPE(*AppFunc_)(const VALTYPE, const VALTYPE))
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
	fHalfStepY = fStepY;
	BoundaryFunction = BoundaryFunc_;
	AppFunction = AppFunc_;
}
#endif

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
	VALTYPE k = fHalfStepX*fHalfStepY; //to dot product

	//first iteration
	r.Laplacian(p,fStepX, fStepY, fHalfStepX,fHalfStepY);
	r.dump("Lap(p)");
	r -= F; //r_{1} = d(p) - F
	r.dump("Lap(p)-F");

	g_1 = r; //запомнили для следующей итерации g_{1}
	rLap.Laplacian(r, fStepX, fStepY, fHalfStepX, fHalfStepY);// d(r_{1})
	rLap.dump("Lap(Lap(p)-F)");
	tau = DotProduct(r, r, k) / DotProduct(rLap, r, k); 
	//std::cout << DotProduct(r, r, k)<<' '<< DotProduct(rLap, r, k)<<' '<< tau << std::endl;
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
		r.Laplacian(p, fStepX,fStepY, fHalfStepX, fHalfStepY);
		r.dump("Lap(r)");
		r -= F; // r_{k}
		r.dump("-F");
		rLap.Laplacian(r, fStepX, fStepY, fHalfStepX, fHalfStepY);
		rLap.dump("Lap(Lap(r))");
		gLap.Laplacian(g_1, fStepX, fStepY, fHalfStepX, fHalfStepY);
		gLap.dump("gLap");
		if (memcmp(rLap.ptrData, gLap.ptrData, gLap.nCols*gLap.nRows * sizeof(VALTYPE)))
		{
			alpha = DotProduct(rLap, g_1, k) / DotProduct(gLap, g_1, k);
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
		gLap.Laplacian(g, fStepX, fStepY, fHalfStepX, fHalfStepY);
		gLap.dump("Lap(g)");
		if (memcmp(gLap.ptrData, g.ptrData, g.nCols*g.nRows * sizeof(VALTYPE)))
		{
			tau = DotProduct(r, g, k) / DotProduct(gLap, g, k);
		}
		else
		{
			tau = 0.0;
		}
		//std::cout << tau << std::endl;
		g *= tau; 
		g.dump("g*tau");

		diff = g.MaxNormDifference();
		p -= g; // p_{k+1} = p_{k} - alpha*g_{k}
		p.dump("p");
		/////////////////////////////////
		float maxV = -std::numeric_limits<VALTYPE>::min();
		VALTYPE * data = p.ptrData;
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
		//std::cout << num_iter <<' '<<diff<<' '<<maxV<< std::endl;
		error = maxV;
		
	}
	solution=p;
	//AgregateSolution();
	std::cout << num_iter << ' ' << error << std::endl;
	//getchar();
	return 0;
}

template <typename VALTYPE>
void DirichletProblem<VALTYPE>::InitSubproblem_mpi()
{
	// Get the number of processes
	MPI_Comm_size(Comm_mpi, &nNumProc);
	// Get the rank of the process
	MPI_Comm_rank(Comm_mpi, &nProcIndex);

	int pow2 = log2(nNumProc);
	int nProcRow = 1<<(pow2/2);
	int nProcCol = 1<<(pow2/2+pow2%2);
	nPointsX = std::floor(VALTYPE(nPointsX) / nProcCol + 0.5);
	nPointsY = std::floor(VALTYPE(nPointsY) / nProcRow + 0.5);
	int nProcX = nProcIndex%nProcCol;
	int nProcY = nProcIndex/nProcCol;
	//VALTYPE segLenX = (x1 - x0) / nProcCol;
	//VALTYPE segLenY = VALTYPE(y1 - y0) / nProcRow;
	VALTYPE segLenX = fStepX * (nPointsX-1);
	VALTYPE segLenY = fStepY * (nPointsY-1);
	x0 = (segLenX+fStepX)*nProcX;
	x1 = x0 + segLenX;
	y0 = (segLenY+fStepY)*nProcY;
	y1 = y0 + segLenY;
	//fStepX = (x1 - x0) / (nPointsX-1);
	//fStepY = (y1 - y0) / (nPointsY-1);
	
	
	bool bHasTopNeighb = nProcY != 0;
	bool bHasBotNeighb = nProcY != nProcRow-1;
	bool bHasLeftNeighb = nProcX != 0;
	bool bHasRightNeighb = nProcX != nProcCol - 1;

	y0 -= bHasTopNeighb ? fStepY : 0;
	nPointsY += bHasTopNeighb ? 1 : 0;
	y1 += bHasBotNeighb ? fStepY : 0;
	nPointsY += bHasBotNeighb ? 1 : 0;
	x0 -= bHasLeftNeighb ? fStepY : 0;
	nPointsX += bHasLeftNeighb ? 1 : 0;
	x1 += bHasRightNeighb ? fStepY : 0;
	nPointsX += bHasRightNeighb ? 1 : 0;
	
	fHalfStepX = fStepX;
	fHalfStepY = fStepY;
}


template <typename VALTYPE>
void DirichletProblem<VALTYPE>::AgregateSolution()
{
	//int nNumOfProc;
	//MPI_Comm_size(Comm_mpi, &nNumOfProc);
	//int pow2 = log2(nNumOfProc);
	//int nProcRow = 1 << (pow2 / 2);
	//int nProcCol = 1 << (pow2 / 2 + pow2 % 2);
	//Grid<VALTYPE> generalSolution(nPointsX_nompi, nPointsY_nompi);
	//generalSolution.Fill(BoundaryFunction, FILL_BOUNDARY, x0_nompi, y0_nompi, fStepX_nompi, fStepY_nompi);
	//float * pDst = generalSolution.ptrData;
	//float * pSrc = solution.ptrData;
	//MPI_Request Req_mpi;
	//for (int i = 0; i < nProcRow; ++i)
	//{
	//	if (!nProcIndex)
	//	{
	//		for (int y = 1; y < nPointsY - 1; ++y)
	//		{
	//			MPI_Gather((void*)(pSrc+y*nPointsX + 1), nPointsX - 2, MPI_FLOAT, pDst+ i*(nPointsX - 1)*(nPointsY - 1) + y*nPointsX + 1, nPointsX - 2, MPI_FLOAT, 0, Comm_mpi);
	//			std::cout << " Success gathered " << nPointsX - 2 << " element to " << y+ i*(nPointsX - 1)*(nPointsY - 1) + y*nPointsX + 1 << " elem " << std::endl;
	//			pSrc += nPointsX;
	//			pDst += nPointsX;
	//		}
	//	}
	//	else if (nProcIndex < (i+1)*nProcCol && nProcIndex >= i*nProcCol)
	//	{
	//		for (int y = 1; y < nPointsY - 1; ++y)
	//		{
	//			MPI_Gather((void*)(pSrc+y*nPointsX+1), nPointsX - 2, MPI_FLOAT, NULL, nPointsX - 2, MPI_FLOAT, 0, Comm_mpi);
	//			std::cout <<nProcIndex<< " Send to gather " << nPointsX - 1 << " element from " << y*nPointsX + 1 << " row " << std::endl;
	//			pSrc += nPointsX;
	//		}
	//	}
	//	else
	//	{
	//		std::cout << i << ' ';
	//		std::cout << nProcIndex << " continue" << std::endl;;
	//		continue;
	//	}
	//	MPI_Barrier(Comm_mpi);
	//}

	std::cout << "Result" << std::endl;
	for (int y = 1; y < nPointsY-1; ++y)
	{
		for (int x = 1; x < nPointsX-1; ++x)
		{
			//std::cout << x0 + fStepX*x << ' ' << y0 + fStepY*y << ' ' << solution.ptrData[y*nPointsX + x] << std::endl;
			std::cout << solution.ptrData[y*nPointsX + x] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl << "Difference" << std::endl;
	for (int y = 1; y < nPointsY-1; ++y)
	{
		for (int x = 1; x < nPointsX-1; ++x)
		{
			//std::cout << x0 + fStepX*x << ' ' << y0 + fStepY*y <<' '
			//	<< fabs(solution.ptrData[y*nPointsX + x] - BoundaryFunction(x0 + fStepX*x, y0 + fStepY*y)) << std::endl;
			std::cout << fabs(solution.ptrData[y*nPointsX + x] - BoundaryFunction(x0 + fStepX*x, y0 + fStepY*y)) << ' ';
		}
		std::cout<<std::endl;
	}
	
}

template <typename VALTYPE>
VALTYPE DirichletProblem<VALTYPE>::SolveMPI()
{
	MPI_Init(NULL, NULL);
	Comm_mpi = MPI_COMM_WORLD;
	InitSubproblem_mpi();
	GridMPI<VALTYPE> p(nPointsY, nPointsX,nProcIndex,nNumProc);
	p.Fill(BoundaryFunction, FILL_BOUNDARY, x0, y0, fStepX, fStepY);
	GridMPI<VALTYPE> F(nPointsY, nPointsX, nProcIndex, nNumProc);
	F.Fill(AppFunction, FILL_ALL, x0, y0, fStepX, fStepY);

	GridMPI<VALTYPE> r(nPointsY, nPointsX, nProcIndex, nNumProc);
	GridMPI<VALTYPE> g(nPointsY, nPointsX, nProcIndex, nNumProc);
	GridMPI<VALTYPE> g_1(nPointsY, nPointsX, nProcIndex, nNumProc);
	GridMPI<VALTYPE> rLap(nPointsY, nPointsX, nProcIndex, nNumProc);
	GridMPI<VALTYPE> gLap(nPointsY, nPointsX, nProcIndex, nNumProc);

	VALTYPE tau = 0;
	VALTYPE alpha = 0;
	VALTYPE diff = 0;
	VALTYPE k = fHalfStepX * fHalfStepY; //to dot product

	r.Laplacian(p, fStepX, fStepY, fHalfStepX, fHalfStepY);
	r -= F; //r_{1} = d(p) - F
	r.Send(1, Comm_mpi);
	r.Refresh(1, Comm_mpi);
	g_1 = r; //запомнили для следующей итерации g_{1}
	
	rLap.Laplacian(r, fStepX, fStepY, fHalfStepX, fHalfStepY);// d(r_{1})
	tau = EstimateTauMPI(r, r, rLap, r, k,Comm_mpi);
	r *= tau;
	p -= r;
	p.Send(1, Comm_mpi);
	p.Refresh(1, Comm_mpi);
	diff = EstimateGlobalDiffMPI(r, Comm_mpi);
	num_iter = 1;
	//other iterations
	while (diff > eps)
	{
		++num_iter;
		r.Laplacian(p, fStepX, fStepY, fHalfStepX, fHalfStepY);
		r -= F; // r_{k}

		r.Send(num_iter, Comm_mpi);
		r.Refresh(num_iter, Comm_mpi);
		rLap.Laplacian(r, fStepX, fStepY, fHalfStepX, fHalfStepY);
		
		g_1.Send(num_iter, Comm_mpi);
		g_1.Refresh(num_iter, Comm_mpi);
		gLap.Laplacian(g_1, fStepX, fStepY, fHalfStepX, fHalfStepY);

		alpha = EstimateTauMPI(rLap, g_1, gLap, g_1, k,Comm_mpi);
		g_1 *= alpha;
		g = r;
		g -= g_1; //g_{k}=r_{k}-alpha*g_{k-1}
		g.Send(num_iter, Comm_mpi);
		g.Refresh(num_iter, Comm_mpi);
		g_1 = g; //запомнили

		gLap.Laplacian(g, fStepX, fStepY, fHalfStepX, fHalfStepY);
		tau = EstimateTauMPI(r, g, gLap, g, k, Comm_mpi);
		//if (!nProcIndex)
		//	std::cout << tau << std::endl;
		g *= tau;
		diff = EstimateGlobalDiffMPI(g, Comm_mpi);;
		p -= g; // p_{k+1} = p_{k} - alpha*g_{k}
		
///////////////////////////////////////
float maxV = -1;
VALTYPE * data = p.ptrData;
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

		error = maxV;
		p.Send(num_iter, Comm_mpi);
		p.Refresh(num_iter, Comm_mpi);
		if (!nProcIndex)
		{
			//std::cout << num_iter << ' ' << error << std::endl;
		}
	}
	//solution = p;
	if (!nProcIndex)
	{
		std::cout << num_iter << ' ' << error << std::endl;
		//AgregateSolution();
		//Sig(Comm_mpi);
		//Wait(Comm_mpi);
	}
	else
	{
		//Wait(Comm_mpi);
		//AgregateSolution();
		//Sig(Comm_mpi);
	}
	MPI_Finalize();
	return 0;

}
/*
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
		r.Laplacian(p, fStepX, fStepY, fHalfStepX, fHalfStepY);
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
		SendBoundaries(r, 1);
		RefreshBoundaries(r, 1);
		rLap.Laplacian(r, fStepX, fStepY, fHalfStepX, fHalfStepY);// d(r_{1})
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
			
			r.Laplacian(p, fStepX, fStepY, fHalfStepX, fHalfStepY);
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
			rLap.Laplacian(r, fStepX, fStepY, fHalfStepX, fHalfStepY);
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
			gLap.Laplacian(g_1, fStepX, fStepY, fHalfStepX, fHalfStepY);
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
			//if (memcmp(rLap.ptrData, gLap.ptrData, gLap.nCols*gLap.nRows * sizeof(VALTYPE)))
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
			gLap.Laplacian(g, fStepX, fStepY, fHalfStepX, fHalfStepY);
		//	gLap.dump("Lap(g)");
			//if (memcmp(gLap.ptrData, g.ptrData, g.nCols*g.nRows * sizeof(VALTYPE)))
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
			VALTYPE * data = p.ptrData;
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
			//std::cout << nProcIndex << ' ' << num_iter << ' ' << diff << ' '<<maxV<< std::endl;
			error = maxV;
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
		//error = diff;
		if (!nProcIndex)
		{
			std::cout << num_iter << ' ' << error << std::endl;
		}
		//std::cout << nProcIndex << " END" << std::endl;
		//Iteration2()
		//SendBoundaries();
		//RefreshBoundaries();
		if (!nProcIndex)
		{
			AgregateSolution();
			Sig(Comm_mpi);
			Wait(Comm_mpi);
		}
		else
		{
			Wait(Comm_mpi);
			AgregateSolution();
			Sig(Comm_mpi);
		}

//	}
	MPI_Finalize();
	return 0;
}

*/