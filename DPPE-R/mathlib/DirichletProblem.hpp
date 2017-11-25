
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

#include<iostream>
#include <iomanip>
//#define IT_LOG
template <typename VALTYPE>
VALTYPE DirichletProblem<VALTYPE>::Solve()
{
	//GridStorage<VALTYPE> storage(nPointsY*nPointsX, 7);
	Grid<VALTYPE> p(nPointsY, nPointsX);
	p.Fill(BoundaryFunction, FILL_BOUNDARY,x0,y0,fStepX);
	p.dump("P");
	Grid<VALTYPE> F(nPointsY, nPointsX);
	F.Fill(AppFunction, FILL_ALL, x0, y0, fStepX);
	F.dump("F");
	Grid<VALTYPE> r(nPointsY, nPointsX);
	Grid<VALTYPE> g(nPointsY, nPointsX);
	Grid<VALTYPE> g_1(nPointsY, nPointsX);
	Grid<VALTYPE> rLap(nPointsY, nPointsX);
	Grid<VALTYPE> gLap(nPointsY, nPointsX);
	VALTYPE tau = 0;
	VALTYPE alpha = 0;
	VALTYPE diff = 0;
	VALTYPE k = fHalfStepX; //to dot product

	//first iteration
	r.Laplacian(p,fStepX,fHalfStepX);
	r.dump("Lap(p)");
#ifdef IT_LOG
	std::cout << std::fixed << std::setprecision(4);
	for (int i = 0; i < p.nRows; ++i)
	{
		for (int j = 0; j < p.nCols; ++j)
		{
			std::cout << (r.ptrData.get())[i*p.nRows + j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
#endif
	r -= F; //r_{1} = d(p) - F
	r.dump("Lap(p)-F");
	g_1 = r; //запомнили для следующей итерации g_{1}
	rLap.Laplacian(r, fStepX, fHalfStepX);// d(r_{1})
	rLap.dump("Lap(Lap(p)-F)");
	tau = DotProduct(r, r, k) / DotProduct(rLap, r, k); 
	//std::cout <<"tau= "<< tau << std::endl;
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
		r.Laplacian(p, fStepX, fHalfStepX);
		r.dump("Lap(r)");
		r -= F; // r_{k}
		r.dump("-F");
		rLap.Laplacian(r, fStepX, fHalfStepX);
		rLap.dump("Lap(Lap(r))");
		gLap.Laplacian(g_1, fStepX, fHalfStepX);
		gLap.dump("gLap");
		if (memcmp(rLap.ptrData.get(), gLap.ptrData.get(), gLap.nCols*gLap.nRows * sizeof(VALTYPE)))
		{
			alpha = DotProduct(rLap, g_1, k) / DotProduct(gLap, g_1, k);
			//std::cout <<"a= "<< alpha << std::endl;
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
		gLap.Laplacian(g, fStepX, fHalfStepX);
		gLap.dump("Lap(g)");
		if (memcmp(gLap.ptrData.get(), g.ptrData.get(), g.nCols*g.nRows * sizeof(VALTYPE)))
		{
			tau = DotProduct(r, g, k) / DotProduct(gLap, g, k);
			//std::cout << "tau= " << tau << std::endl;
		}
		else
		{
			tau = 0.0;
		}
		g *= tau; 
		g.dump("g*tau");
		diff = g.MaxNormDifference();
		std::cout << num_iter <<' '<<diff<< std::endl;
		p -= g; // p_{k+1} = p_{k} - alpha*g_{k}
		p.dump("p");
#ifdef IT_LOG
		for (int i = 0; i < p.nRows; ++i)
		{
			for (int j = 0; j < p.nCols; ++j)
			{
				std::cout << (p.ptrData.get())[i*p.nRows + j] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
#endif
	}
	solution=p;
	error = diff;
	return 0;
}

template <typename VALTYPE>
VALTYPE DirichletProblem<VALTYPE>::SolveMPI()
{
	return 0;
}

template <typename VALTYPE>
void DirichletProblem<VALTYPE>::SetPrecision(const VALTYPE eps_)
{
	eps = eps_;
}