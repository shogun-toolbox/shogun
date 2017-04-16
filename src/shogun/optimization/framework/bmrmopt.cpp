#include <shogun/optimization/framework/optdefines.h>
#include <shogun/optimization/framework/bmrmopt.h>
using namespace shogun;

uint32_t BSize;
float64_t* Hs;
const float64_t* get_col1(uint32_t i)
{
	return(&Hs[BSize*i]);
}
const float64_t* (*get_c)(uint32_t) = &get_col1;

bmrmOptimizer::bmrmOptimizer()
{
	fx = func();
	cx = constraints();
	cps = CP();
	bmrm = BmrmStatistics();
	W = NULL;
	prevW = NULL;
	mach = NULL;
	BufSize = 0;
	cleanICP = false;
	verbose = false;
	cleanAfter = 0;
	lambda = 0;
}

int bmrmOptimizer::setup(		
		CDualLibQPBMSOSVM  *machine,
		float64_t*       Wt,
		float64_t        TolRel,
		float64_t        TolAbs,
		float64_t        _lambda,
		uint32_t         _BufSize,
		bool             cICP,
		uint32_t         cAfter,
		bool             vrbose)
{
	/* This includes dependancy from machine 
	 * Try to remove it.
	 * */
	mach = machine;
	BufSize = _BufSize;
	histSize = BufSize;
	BSize = BufSize;
	histSize = BufSize;
	lambda = _lambda;
	cleanICP = cICP;
	cleanAfter = cAfter;
	verbose = vrbose;
	CStructuredModel* model=machine->get_model();
	uint32_t nDim=model->get_dim();
	prevW = (float64_t*) BMRM_CALLOC( nDim, float64_t);
	int a = fx.init(nDim, BufSize), b= cx.init(TolRel, TolAbs, nDim, BufSize), c= cps.init(BufSize, nDim, BufSize);
	if( a != 0 || b != 0 || c != 0 || prevW == NULL)
	{
		/* Don't forget to free resources */
		return -1;
	}
	
	Hs = fx.H;
	W = Wt;
	/* Setting up bmrm options */
	bmrm.hist_Fp = SGVector< float64_t >(BufSize);
	bmrm.hist_Fd = SGVector< float64_t >(BufSize);
	bmrm.hist_wdist = SGVector< float64_t >(BufSize);
	bmrm.nCP=0;
	bmrm.nIter=0;
	bmrm.exitflag=0;
	float64_t R=machine->risk(fx.subgrad, W);
	fx.f[0] = -R;
	bmrm.Fp=R;
	bmrm.Fd=-BMRM_PLUS_INF;
	bmrm.hist_Fp[0]=bmrm.Fp;
	bmrm.hist_Fd[0]=bmrm.Fd;
	bmrm.hist_wdist[0]=0.0;
	ASSERT(bmrm.nCP<BufSize);
	
	BMRM_MEMCPY(fx.A, fx.subgrad, nDim*sizeof(float64_t));
	cps.map[0]=false;
	(cps.cplist)->address=&(fx.A[0]);
	(cps.cplist)->idx=0;
	(cps.cplist)->prev=NULL;
	(cps.cplist)->next=NULL;
	*(cps.head)=cps.cplist;
	*(cps.tail)=cps.cplist;
}

void bmrmOptimizer::computeAndUpdate(bool tune_alpha)
{
	/*
	 * Steps in computation :
	 * 1. Updating Hessian
	 * 2. Getting the current optimal point by solving QP.
	 * 3. Updating the current weight.
	 * 4. Processing Inactive Cutting Planes
	 * 5. Risk and Subgradient Computation
	 * 6. Repeat if outside tolerance
	 * 
	 * Optional Steps in other bundle methods : 
	 * 1. Alpha Tuning
	 */
	 updateHessian();
	 libqp_state_T qpsolve = solveQP(tune_alpha);
	 updateWeights();
	 computeRandS(qpsolve);
	 removeICPs();
	 return ;
}

void bmrmOptimizer::updateHessian()
{
		/* See things about _lambda */
		bmrm_ll* cp_ptr = NULL;
		floatmax_t rsum = 0.0;
		bmrm_ll* hd = *(cps.head);
		bmrm_ll* tl = *(cps.tail);
		float64_t *A1 = NULL, *A2 = NULL;
		uint32_t nDim = fx.dim;
		bmrm.nIter++;
	
		if (bmrm.nCP>0)
		{
			A1=cps.get_cutting_plane(tl);
			cp_ptr=hd;

			for (uint32_t i=0; i<bmrm.nCP; ++i)
			{
				A1=cps.get_cutting_plane(cp_ptr);
				cp_ptr=cp_ptr->next;
				rsum= SGVector<float64_t>::dot(A1, A2, nDim);

				fx.H[BMRM_INDEX(bmrm.nCP, i, BufSize)]
					= fx.H[BMRM_INDEX(i, bmrm.nCP, BufSize)]
					= rsum/lambda;
			}
		}

		A2=cps.get_cutting_plane(tl);
		rsum = SGVector<float64_t>::dot(A2, A2, nDim);

		fx.H[BMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)]=rsum/lambda;
		fx.diag_H[bmrm.nCP]=fx.H[BMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)];
		cx.I[bmrm.nCP]=1;

		fx.x[bmrm.nCP]=0.0; // [beta; 0] -> what is beta?
		bmrm.nCP++;
		ASSERT(bmrm.nCP<BufSize);
		return ;
}

libqp_state_T bmrmOptimizer::solveQP(bool tune_alpha)
{
	libqp_state_T qp_exitflag = {0, 0, 0, 0};
		/* call QP solver */
		/** Modify the implementation of get_col **/
	qp_exitflag=libqp_splx_solver(get_c, fx.diag_H, fx.f, cx.UB, cx.I, cx.S, fx.x,
				bmrm.nCP, 0xFFFFFFFF, 0.0, 1e-9, -BMRM_PLUS_INF, 0);
	bmrm.qp_exitflag = qp_exitflag.exitflag;
	bmrm.nzA = 0;
	return qp_exitflag;
}

void bmrmOptimizer::updateWeights()
{
	/* This includes ICPs Cleanup */
	uint32_t nDim = fx.dim;
	float64_t *A1 = NULL;
	bmrm_ll* cp_ptr = NULL;
	for (uint32_t aaa=0; aaa<bmrm.nCP; ++aaa)
		{
			if (fx.x[aaa]>0.0)
			{
				++bmrm.nzA;
				cps.ICPcounter[aaa]=0;
			}
			else
			{
				/* It may be possible that the values are less than zero but this equation
				 * tracks which planes may not be needed anymore.
				 */
				cps.ICPcounter[aaa]+=1;
			}
		}

		/* W update */
		memset(W, 0, sizeof(float64_t)*nDim);
		cp_ptr=*(cps.head);
		for (uint32_t j=0; j<bmrm.nCP; ++j)
		{
			A1=cps.get_cutting_plane(cp_ptr);
			cp_ptr=cp_ptr->next;
			SGVector<float64_t>::vec1_plus_scalar_times_vec2(W, -fx.x[j]/lambda, A1, nDim);
		}
	return ;
}

void bmrmOptimizer::computeRandS(libqp_state_T qps)
{
		/** Store these parameters - machine , lambda , prevW , histSize **/
		/* Also includes keeping history of the weights and functions */
		float64_t R = mach->risk(fx.subgrad, W);
		cps.add_cutting_plane(fx.A,
				cps.find_free_idx(cps.map, BufSize), fx.subgrad);

		float64_t sq_norm_W=SGVector<float64_t>::dot(W, W, fx.dim);
		fx.f[bmrm.nCP]=SGVector<float64_t>::dot(fx.subgrad, W, fx.dim) - R;

		float64_t sq_norm_Wdiff=0.0;
		for (uint32_t j=0; j<fx.dim; ++j)
		{
			sq_norm_Wdiff+=(W[j]-prevW[j])*(W[j]-prevW[j]);
		}

		bmrm.Fp=R+0.5*lambda*sq_norm_W;
		bmrm.Fd=-qps.QP;
		float64_t wdist=CMath::sqrt(sq_norm_Wdiff);

		/* Stopping conditions for the QP */
		if (bmrm.Fp - bmrm.Fd <= cx.TolRel*BMRM_ABS(bmrm.Fp))
			bmrm.exitflag=1; //Replace with appropriate return conditions

		if (bmrm.Fp - bmrm.Fd <= cx.TolAbs)
			bmrm.exitflag=2;

/** Include verbose output later **/
		/* Verbose output 
		if (verbose)
			SG_SPRINT("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, nCP=%d, nzA=%d, QPexitflag=%d\n",
					bmrm.nIter, tstop-tstart, bmrm.Fp, bmrm.Fd, bmrm.Fp-bmrm.Fd,
					(bmrm.Fp-bmrm.Fd)/bmrm.Fp, R, bmrm.nCP, bmrm.nzA, qp_exitflag.exitflag);
		*/
		
		if (bmrm.nIter >= histSize)
		{
			histSize += BufSize;
			bmrm.hist_Fp.resize_vector(histSize);
			bmrm.hist_Fd.resize_vector(histSize);
			bmrm.hist_wdist.resize_vector(histSize);
		}

		/* Keep Fp, Fd and w_dist history */
		ASSERT(bmrm.nIter < histSize);
		bmrm.hist_Fp[bmrm.nIter]=bmrm.Fp;
		bmrm.hist_Fd[bmrm.nIter]=bmrm.Fd;
		bmrm.hist_wdist[bmrm.nIter]=wdist;

		/* keep W (for wdist history track) */
		BMRM_MEMCPY(prevW, W, (fx.dim)*sizeof(float64_t));
		return ;
}

void bmrmOptimizer::removeICPs()
{
	/** CleanICP **/
	if (cleanICP)
	{
			cps.clean_icp(bmrm, fx.H, fx.diag_H, fx.x, cleanAfter, fx.f, cx.I);
			ASSERT(bmrm.nCP<BufSize);
	}
	
	if (bmrm.nCP+1 >= BufSize)
		bmrm.exitflag=-1;
}

void bmrmOptimizer::cleanup()
{
	ASSERT(bmrm.nIter+1 <= histSize);
	bmrm.hist_Fp.resize_vector(bmrm.nIter+1);
	bmrm.hist_Fd.resize_vector(bmrm.nIter+1);
	bmrm.hist_wdist.resize_vector(bmrm.nIter+1);

	bmrm_ll* cp_ptr=*(cps.head);
	bmrm_ll* cp_ptr2 = NULL;
	while(cp_ptr!=NULL)
	{
		cp_ptr2=cp_ptr;
		cp_ptr=cp_ptr->next;
		BMRM_FREE(cp_ptr2);
		cp_ptr2=NULL;
	}

	cps.cplist=NULL;
	fx.cleanup();
	cx.cleanup();
	cps.cleanup();
	BMRM_FREE(W);
	BMRM_FREE(prevW);
	//SG_UNREF(model);
	/** Put In all the other parameters **/
}


int bmrmOptimizer::returnOptima(
		CDualLibQPBMSOSVM  *machine,
		float64_t*       Wt,
		float64_t        TolRel,
		float64_t        TolAbs,
		float64_t        _lambda,
		uint32_t         _BufSize,
		bool             cICP,
		uint32_t         cAfter,
		bool             vrbose)
{
	if(setup(machine, Wt, TolRel, TolAbs, _lambda, _BufSize, cICP, cAfter, vrbose) < 0)
		return -1;

	while(bmrm.exitflag==0)
	{
		computeAndUpdate(false);	
	}
	cleanup();
	return 0;
}

