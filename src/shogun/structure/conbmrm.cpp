/*
* This file contains all the methods for different convex bundle solvers.
*/

#include <shogun/structure/consolver.h>
#include <shogun/lib/external/libqp.h>
#include <shogun/lib/Time.h>
#include <shogun/io/SGIO.h>
#include <shogun/multiclass/GMNPLib.h>
#include <vector>

namespace shogun {
static const uint32_t QPSolverMaxIter=0xFFFFFFFF;
static const float64_t epsilon=0.0;
static uint32_t solver_type;
static float64_t *H, *H2;
static uint32_t BufSize;

void add_cutting_plane(
		bmrm_ll**	tail,
		bool*		map,
		float64_t*	A,
		uint32_t	free_idx,
		float64_t*	cp_data,
		uint32_t	dim)
{
	ASSERT(map[free_idx])

	LIBBMRM_MEMCPY(A+free_idx*dim, cp_data, dim*sizeof(float64_t));
	map[free_idx]=false;

	bmrm_ll *cp=(bmrm_ll*)LIBBMRM_CALLOC(1, bmrm_ll);

	if (cp==NULL)
	{
		SG_SERROR("Out of memory.\n")
		return;
	}

	cp->address=A+(free_idx*dim);
	cp->prev=*tail;
	cp->next=NULL;
	cp->idx=free_idx;
	(*tail)->next=cp;
	*tail=cp;
}

void remove_cutting_plane(
		bmrm_ll**	head,
		bmrm_ll**	tail,
		bool*		map,
		float64_t*	icp)
{
	bmrm_ll *cp_list_ptr=*head;

	while(cp_list_ptr->address != icp)
	{
		cp_list_ptr=cp_list_ptr->next;
	}

	if (cp_list_ptr==*head)
	{
		*head=(*head)->next;
		cp_list_ptr->next->prev=NULL;
	}
	else if (cp_list_ptr==*tail)
	{
		*tail=(*tail)->prev;
		cp_list_ptr->prev->next=NULL;
	}
	else
	{
		cp_list_ptr->prev->next=cp_list_ptr->next;
		cp_list_ptr->next->prev=cp_list_ptr->prev;
	}

	map[cp_list_ptr->idx]=true;
	LIBBMRM_FREE(cp_list_ptr);
}

void clean_icp(ICP_stats* icp_stats,
		BmrmStatistics& bmrm,
		bmrm_ll** head,
		bmrm_ll** tail,
		float64_t*& Hmat,
		float64_t*& diag_H,
		float64_t*& beta,
		bool*& map,
		uint32_t cleanAfter,
		float64_t*& b,
		uint32_t*& I,
		uint32_t cp_models
		)
{
	/* find ICP */
	uint32_t cntICP=0;
	uint32_t cntACP=0;
	bmrm_ll* cp_ptr=*head;
	uint32_t tmp_idx=0;

	while (cp_ptr != *tail)
	{
		if (icp_stats->ICPcounter[tmp_idx++]>=cleanAfter)
		{
			icp_stats->ICPs[cntICP++]=cp_ptr->address;
		}
		else
		{
			icp_stats->ACPs[cntACP++]=tmp_idx-1;
		}

		cp_ptr=cp_ptr->next;
	}

	/* do ICP removal */
	if (cntICP > 0)
	{
		uint32_t nCP_new=solver.nCP-cntICP;

		for (uint32_t i=0; i<cntICP; ++i)
		{
			tmp_idx=0;
			cp_ptr=*head;

			while(cp_ptr->address != icp_stats->ICPs[i])
			{
				cp_ptr=cp_ptr->next;
				tmp_idx++;
			}

			remove_cutting_plane(head, tail, map, icp_stats->ICPs[i]);

			LIBBMRM_MEMMOVE(b+tmp_idx, b+tmp_idx+1,
					(solver.nCP+cp_models-tmp_idx)*sizeof(float64_t));
			LIBBMRM_MEMMOVE(beta+tmp_idx, beta+tmp_idx+1,
					(solver.nCP-tmp_idx)*sizeof(float64_t));
			LIBBMRM_MEMMOVE(diag_H+tmp_idx, diag_H+tmp_idx+1,
					(solver.nCP-tmp_idx)*sizeof(float64_t));
			LIBBMRM_MEMMOVE(I+tmp_idx, I+tmp_idx+1,
					(solver.nCP-tmp_idx)*sizeof(uint32_t));
			LIBBMRM_MEMMOVE(icp_stats->ICPcounter+tmp_idx, icp_stats->ICPcounter+tmp_idx+1,
					(solver.nCP-tmp_idx)*sizeof(uint32_t));
		}

		/* H */
		for (uint32_t i=0; i < nCP_new; ++i)
		{
			for (uint32_t j=0; j < nCP_new; ++j)
			{
				icp_stats->H_buff[LIBBMRM_INDEX(i, j, icp_stats->maxCPs)]=
					Hmat[LIBBMRM_INDEX(icp_stats->ACPs[i], icp_stats->ACPs[j], icp_stats->maxCPs)];
			}
		}

		for (uint32_t i=0; i<nCP_new; ++i)
			for (uint32_t j=0; j<nCP_new; ++j)
				Hmat[LIBBMRM_INDEX(i, j, icp_stats->maxCPs)]=
					icp_stats->H_buff[LIBBMRM_INDEX(i, j, icp_stats->maxCPs)];

		solver.nCP=nCP_new;
	}
}

/*----------------------------------------------------------------------
  Returns pointer at i-th column of Hessian matrix.
  ----------------------------------------------------------------------*/
static const float64_t *get_col( uint32_t i)
{
	if(solver_type == PROXIMAL_POINT_BMRM || solver_type == PROXIMAL_P_BMRM)
		return( &H2[ BufSize*i ] );
	else
		return( &H[ BufSize*i ] );
}

BmrmStatistics con_bmrm_solver(
		CDualLibQPBMSOSVM  *machine,
		float64_t*       W,
		float64_t        TolRel,
		float64_t        TolAbs,
		float64_t        _lambda,
		uint32_t         _BufSize,
		bool             cleanICP,
		uint32_t         cleanAfter,
		float64_t        K,
		uint32_t         Tmax,
		uint32_t		 cp_models,
		bool             verbose,
		uint32_t		 solvertype)
{
	BmrmStatistics solver;
	solver_type = solvertype;
	libqp_state_T qp_exitflag={0, 0, 0, 0}, qp_exitflag_good = {0,0,0,0};
	float64_t *b, *beta, *beta_good, *beta_start, *diag_H, *diag_H2, *prevW;
	float64_t R, *Rt, *subgrad, **subgrad_t, *A, QPSolverTolRel, *C, Cx;
	float64_t *wt, alpha, alpha_start, alpha_good, Fd_alpha0;
	float64_t lastFp, wdist, gamma;
	QPSolverTolRel=1e-9;
	
	floatmax_t rsum, sq_norm_W, sq_norm_Wdiff, sq_norm_prevW, eps;
	uint32_t *I, *I2, *I_start, *I_good;
	uint8_t Sx, *S = NULL;
	TMultipleCPinfo **info = NULL;
	CStructuredModel* model=machine->get_model();
	uint32_t nDim=model->get_dim();
	uint32_t to=0, N=0, cp_i=0;
	CSOSVMHelper* helper = NULL;
	uint32_t qp_cnt;
	CTime ttime;
	float64_t tstart, tstop;
	tstart=ttime.cur_time_diff(false);
	
	bmrm_ll *CPList_head, *CPList_tail, *cp_ptr, *cp_ptr2, *cp_list=NULL;
	float64_t *A_1=NULL, *A_2=NULL;
	bool *map=NULL, tuneAlpha, flag, alphaChanged, isThereGoodSolution;
	
	switch(solvertype)
	{
		case BBRM: {	BufSize = _BufSize;
						Cx = 1.0;
						wDist = 0.0;
						sq_norm_Wdiff = 0.0;
						Sx = 1;
						solver_type = BMRM;
						break;
					}
		case PPBMRM : {	BufSize = _BufSize;
						Cx = 1.0;
						alpha_good = 0.0;
						Fd_alpha0=0.0;
						gamma=0.0;
						Sx = 1;
						alpha = 0.0;
						tuneAlpha=true;
						flag=true;
						alphaChanged=false;
						isThereGoodSolution=false;
						solver_type = PPBM;
						break;
					}
		case P3BMRM : {	BufSize = _BufSize*cp_models;
						C = NULL;
						gamma = 0.0;
						alpha_good = 0.0;
						Fd_alpha0=0.0;
						tuneAlpha=true;
						flag=true;
						alphaChanged=false;
						isThereGoodSolution=false;
						alpha = 0.0;
						qp_cnt = 0.0;
						solver_type = P3BM;
						break;
					}
		default : 	{	goto exitpath;
						break;
					}
	}
	
	H=NULL;
	b=NULL;
	beta=NULL;
	A=NULL;
	subgrad = NULL;
	subgrad_t=NULL;
	diag_H=NULL;
	I=NULL;
	prevW=NULL;
	wt=NULL;
	diag_H2=NULL;
	b2=NULL;
	I2=NULL;
	H2=NULL;
	I_good=NULL;
	I_start=NULL;
	beta_start=NULL;
	beta_good=NULL;
	C=NULL;
	Rt=NULL;
	
	H= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, float64_t);
	A= (float64_t*) LIBBMRM_CALLOC(nDim*BufSize, float64_t);
	b= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
	beta= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
	diag_H= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
	I= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
	cp_list= (bmrm_ll*) LIBBMRM_CALLOC(1, bmrm_ll);
	prevW= (float64_t*) LIBBMRM_CALLOC(nDim, float64_t);
	map= (bool*) LIBBMRM_CALLOC(BufSize, bool);
	
	ICP_stats icp_stats;
	icp_stats.maxCPs = BufSize;
	icp_stats.ICPcounter= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
	icp_stats.ICPs= (float64_t**) LIBBMRM_CALLOC(BufSize, float64_t*);
	icp_stats.ACPs= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
	
	if (H==NULL || A==NULL || b==NULL || beta==NULL || diag_H==NULL 
		|| I==NULL || cp_list==NULL || prevW==NULL || icp_stats.ICPcounter==NULL
		|| icp_stats.ICPs==NULL || icp_stats.ACPs==NULL || icp_stats.H_buff==NULL)
	{
		solver.exitflag=-2;
		goto exitpath;
	}

	memset( (bool*) map, true, BufSize);
	/*Actual BMRM solving process starts here*/
	switch(solvertype)
	{
		case BMRM : {
						subgrad= (float64_t*) LIBBMRM_CALLOC(nDim, float64_t);
						if (subgrad==NULL)
						{
							solver.exitflag=-2;
							goto exitpath-BMRM;
						}
						
						solver.hist_Fp = SGVector< float64_t >(BufSize);
						solver.hist_Fd = SGVector< float64_t >(BufSize);
						solver.hist_wdist = SGVector< float64_t >(BufSize);
					
						/* Initial solution */
						R=machine->risk(subgrad, W);
					
						solver.nCP=0;
						solver.nIter=0;
						solver.exitflag=0;
					
						b[0]=-R;
					
						/* Cutting plane auxiliary double linked list */
					
						LIBBMRM_MEMCPY(A, subgrad, nDim*sizeof(float64_t));
						map[0]=false;
						cp_list->address=&A[0];
						cp_list->idx=0;
						cp_list->prev=NULL;
						cp_list->next=NULL;
						CPList_head=cp_list;
						CPList_tail=cp_list;
					
						/* Compute initial value of Fp, Fd, assuming that W is zero vector */
					
						sq_norm_W=0;
						solver.Fp=R+0.5*_lambda*sq_norm_W;
						solver.Fd=-LIBBMRM_PLUS_INF;
					
						tstop=ttime.cur_time_diff(false);
					
						/* Verbose output */
					
						if (verbose)
							SG_SDEBUG("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, R=%lf\n",
									solver.nIter, tstop-tstart, solver.Fp, solver.Fd, R);
					
						/* store Fp, Fd and wdist history */
						solver.hist_Fp[0]=solver.Fp;
						solver.hist_Fd[0]=solver.Fd;
						solver.hist_wdist[0]=0.0;
					
						if (verbose)
							helper = machine->get_helper();
					
						/* main loop */
					
						while (solver.exitflag==0)
						{
							tstart=ttime.cur_time_diff(false);
							solver.nIter++;
					
							/* Update H */
					
							if (solver.nCP>0)
							{
								A_2=get_cutting_plane(CPList_tail);
								cp_ptr=CPList_head;
					
								for (uint32_t i=0; i<solver.nCP; ++i)
								{
									A_1=get_cutting_plane(cp_ptr);
									cp_ptr=cp_ptr->next;
									rsum= SGVector<float64_t>::dot(A_1, A_2, nDim);
					
									H[LIBBMRM_INDEX(solver.nCP, i, BufSize)]
										= H[LIBBMRM_INDEX(i, solver.nCP, BufSize)]
										= rsum/_lambda;
								}
							}
					
							A_2=get_cutting_plane(CPList_tail);
							rsum = SGVector<float64_t>::dot(A_2, A_2, nDim);
					
							H[LIBBMRM_INDEX(solver.nCP, solver.nCP, BufSize)]=rsum/_lambda;
					
							diag_H[solver.nCP]=H[LIBBMRM_INDEX(solver.nCP, solver.nCP, BufSize)];
							I[solver.nCP]=1;
					
							solver.nCP++;
							beta[solver.nCP]=0.0; // [beta; 0]
					
					#if 0
							/* TODO: scaling...*/
							float64_t scale = SGVector<float64_t>::max(diag_H, BufSize)/(1000.0*_lambda);
							SGVector<float64_t> sb(solver.nCP);
							sb.zero();
							sb.vec1_plus_scalar_times_vec2(sb.vector, 1/scale, b, solver.nCP);
					
							SGVector<float64_t> sh(solver.nCP);
							sh.zero();
							sb.vec1_plus_scalar_times_vec2(sh.vector, 1/scale, diag_H, solver.nCP);
					
							qp_exitflag =
								libqp_splx_solver(&get_col, sh.vector, sb.vector, &C, I, &S, beta,
									solver.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
					#else
							/* call QP solver */
							qp_exitflag=libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, beta,
									solver.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
					#endif
					
							solver.qp_exitflag=qp_exitflag.exitflag;
					
							/* Update ICPcounter (add one to unused and reset used)
							 * + compute number of active CPs */
							solver.nzA=0;
					
							for (uint32_t aaa=0; aaa<solver.nCP; ++aaa)
							{
								if (beta[aaa]>epsilon)
								{
									++solver.nzA;
									icp_stats.ICPcounter[aaa]=0;
								}
								else
								{
									icp_stats.ICPcounter[aaa]+=1;
								}
							}
					
							/* W update */
							memset(W, 0, sizeof(float64_t)*nDim);
							cp_ptr=CPList_head;
							for (uint32_t j=0; j<solver.nCP; ++j)
							{
								A_1=get_cutting_plane(cp_ptr);
								cp_ptr=cp_ptr->next;
								SGVector<float64_t>::vec1_plus_scalar_times_vec2(W, -beta[j]/_lambda, A_1, nDim);
							}
					
							/* risk and subgradient computation */
							R = machine->risk(subgrad, W);
							add_cutting_plane(&CPList_tail, map, A,
									find_free_idx(map, BufSize), subgrad, nDim);
					
							sq_norm_W=SGVector<float64_t>::dot(W, W, nDim);
							b[solver.nCP]=SGVector<float64_t>::dot(subgrad, W, nDim) - R;
					
							sq_norm_Wdiff=0.0;
							for (uint32_t j=0; j<nDim; ++j)
							{
								sq_norm_Wdiff+=(W[j]-prevW[j])*(W[j]-prevW[j]);
							}
					
							solver.Fp=R+0.5*_lambda*sq_norm_W;
							solver.Fd=-qp_exitflag.QP;
							wdist=CMath::sqrt(sq_norm_Wdiff);
					
							/* Stopping conditions */
					
							if (solver.Fp - solver.Fd <= TolRel*LIBBMRM_ABS(solver.Fp))
								solver.exitflag=1;
					
							if (solver.Fp - solver.Fd <= TolAbs)
								solver.exitflag=2;
					
							if (solver.nCP >= BufSize)
								solver.exitflag=-1;
					
							tstop=ttime.cur_time_diff(false);
					
							/* Verbose output */
					
							if (verbose)
								SG_SDEBUG("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, nCP=%d, nzA=%d, QPexitflag=%d\n",
										solver.nIter, tstop-tstart, solver.Fp, solver.Fd, solver.Fp-solver.Fd,
										(solver.Fp-solver.Fd)/solver.Fp, R, solver.nCP, solver.nzA, qp_exitflag.exitflag);
					
							/* Keep Fp, Fd and w_dist history */
							solver.hist_Fp[solver.nIter]=solver.Fp;
							solver.hist_Fd[solver.nIter]=solver.Fd;
							solver.hist_wdist[solver.nIter]=wdist;
					
							/* Check size of Buffer */
					
							if (solver.nCP>=BufSize)
							{
								solver.exitflag=-2;
								SG_SERROR("Buffer exceeded.\n")
							}
					
							/* keep W (for wdist history track) */
							LIBBMRM_MEMCPY(prevW, W, nDim*sizeof(float64_t));
					
							/* Inactive Cutting Planes (ICP) removal */
							if (cleanICP)
							{
								clean_icp(&icp_stats, bmrm, &CPList_head, &CPList_tail, H, diag_H, beta, map, cleanAfter, b, I);
							}
					
							/* Debug: compute objective and training error */
							if (verbose)
							{
								SGVector<float64_t> w_debug(W, nDim, false);
								float64_t primal = CSOSVMHelper::primal_objective(w_debug, model, _lambda);
								float64_t train_error = CSOSVMHelper::average_loss(w_debug, model);
								helper->add_debug_info(primal, solver.nIter, train_error);
							}
						} /* end of main loop */
					
						if (verbose)
						{
							helper->terminate();
							SG_UNREF(helper);
						}
					
						solver.hist_Fp.resize_vector(solver.nIter);
						solver.hist_Fd.resize_vector(solver.nIter);
						solver.hist_wdist.resize_vector(solver.nIter);
					
						cp_ptr=CPList_head;
					
						while(cp_ptr!=NULL)
						{
							cp_ptr2=cp_ptr;
							cp_ptr=cp_ptr->next;
							LIBBMRM_FREE(cp_ptr2);
							cp_ptr2=NULL;
						}
					
						cp_list=NULL;
						exitpath-BMRM :
							LIBBMRM_FREE(subgrad);
							goto exitpath;
						
						break;
		}
		
		case P3BMRM : {	subgrad_t= (float64_t**) LIBBMRM_CALLOC(cp_models, float64_t*);
						Rt= (float64_t*) LIBBMRM_CALLOC(cp_models, float64_t);
						wt= (float64_t*) LIBBMRM_CALLOC(nDim, float64_t);
						C= (float64_t*) LIBBMRM_CALLOC(cp_models, float64_t);
						S= (uint8_t*) LIBBMRM_CALLOC(cp_models, uint8_t);
						info= (TMultipleCPinfo**) LIBBMRM_CALLOC(cp_models, TMultipleCPinfo*);
						
						CFeatures* features = model->get_features();
						int32_t num_feats = features->get_num_vectors();
						SG_UNREF(features);
						
						/* Temporary buffers */
						beta_start= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
						beta_good= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
						b2= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
						diag_H2= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
						H2= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, float64_t);
						I_start= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
						I_good= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
						I2= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
						
						if(subgrad_t == NULL || Rt == NULL || wt == NULL || C == NULL ||
						S == NULL || info == NULL || beta_start == NULL || beta_good == NULL || b2 == NULL ||
						  diag_H2 == NULL || H2 == NULL || I_start == NULL || I_good == NULL || I2 == NULL)
						  {
								solver.exitflag = -2;
								goto exitpath-P3BM;
						  }
						  
						solver.hist_Fp.resize_vector(BufSize);
						solver.hist_Fd.resize_vector(BufSize);
						solver.hist_wdist.resize_vector(BufSize);
					
						/* Iinitial solution */
						Rt[0] = machine->risk(subgrad_t[0], W, info[0]);
					
						solver.nCP=0;
						solver.nIter=0;
						solver.exitflag=0;
					
						b[0]=-Rt[0];
					
						/* Cutting plane auxiliary double linked list */
						LIBBMRM_MEMCPY(A, subgrad_t[0], nDim*sizeof(float64_t));
						map[0]=false;
						cp_list->address=&A[0];
						cp_list->idx=0;
						cp_list->prev=NULL;
						cp_list->next=NULL;
						CPList_head=cp_list;
						CPList_tail=cp_list;
					
						for (uint32_t p=1; p<cp_models; ++p)
						{
							Rt[p] = machine->risk(subgrad_t[p], W, info[p]);
							b[p]=SGVector<float64_t>::dot(subgrad_t[p], W, nDim) - Rt[p];
							add_cutting_plane(&CPList_tail, map, A, find_free_idx(map, BufSize), subgrad_t[p], nDim);
						}
					
						/* Compute initial value of Fp, Fd, assuming that W is zero vector */
						R=0.0;
					
						for (uint32_t p=0; p<cp_models; ++p)
							R+=Rt[p];
					
						sq_norm_W=SGVector<float64_t>::dot(W, W, nDim);
						sq_norm_Wdiff=0.0;
					
						for (uint32_t j=0; j<nDim; ++j)
						{
							sq_norm_Wdiff+=(W[j]-prevW[j])*(W[j]-prevW[j]);
						}
					
						wdist=CMath::sqrt(sq_norm_Wdiff);
					
						solver.Fp=R+0.5*_lambda*sq_norm_W + alpha*sq_norm_Wdiff;
						solver.Fd=-LIBBMRM_PLUS_INF;
						lastFp=solver.Fp;
					
						/* if there is initial W, then set K to be 0.01 times its norm */
						K = (sq_norm_W == 0.0) ? 0.4 : 0.01*CMath::sqrt(sq_norm_W);
					
						LIBBMRM_MEMCPY(prevW, W, nDim*sizeof(float64_t));
					
						tstop=ttime.cur_time_diff(false);
					
						/* Keep history of Fp, Fd, and wdist */
						solver.hist_Fp[0]=solver.Fp;
						solver.hist_Fd[0]=solver.Fd;
						solver.hist_wdist[0]=wdist;
					
						/* Verbose output */
						if (verbose)
							SG_SDEBUG("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, R=%lf, K=%lf, CPmodels=%d\n",
									solver.nIter, tstop-tstart, solver.Fp, solver.Fd, R, K, cp_models);
					
						if (verbose)
							helper = machine->get_helper();
					
						/* main loop */
						while (solver.exitflag==0)
						{
							tstart=ttime.cur_time_diff(false);
							solver.nIter++;
					
							/* Update H */
							if (solver.nIter==1)
							{
								cp_ptr=CPList_head;
					
								for (cp_i=0; cp_i<cp_models; ++cp_i)  /* for all cutting planes */
								{
									A_1=get_cutting_plane(cp_ptr);
					
									for (uint32_t p=0; p<cp_models; ++p)
									{
										rsum=SGVector<float64_t>::dot(A_1, subgrad_t[p], nDim);
					
										H[LIBBMRM_INDEX(p, cp_i, BufSize)]=rsum;
									}
					
									cp_ptr=cp_ptr->next;
								}
							}
							else
							{
								cp_ptr=CPList_head;
					
								for (cp_i=0; cp_i<solver.nCP+cp_models; ++cp_i)  /* for all cutting planes */
								{
									A_1=get_cutting_plane(cp_ptr);
					
									for (uint32_t p=0; p<cp_models; ++p)
									{
										rsum=SGVector<float64_t>::dot(A_1, subgrad_t[p], nDim);
					
										H[LIBBMRM_INDEX(solver.nCP+p, cp_i, BufSize)]=rsum;
									}
					
									cp_ptr=cp_ptr->next;
								}
					
								for (uint32_t i=0; i<solver.nCP; ++i)
									for (uint32_t j=0; j<cp_models; ++j)
										H[LIBBMRM_INDEX(i, solver.nCP+j, BufSize)]=
											H[LIBBMRM_INDEX(solver.nCP+j, i, BufSize)];
							}
					
							for (uint32_t p=0; p<cp_models; ++p)
								diag_H[solver.nCP+p]=H[LIBBMRM_INDEX(solver.nCP+p, solver.nCP+p, BufSize)];
					
							solver.nCP+=cp_models;
					
							/* tune alpha cycle */
							/* ------------------------------------------------------------------------ */
							flag=true;
							isThereGoodSolution=false;
					
							for (uint32_t p=0; p<cp_models; ++p)
							{
								I[solver.nCP-cp_models+p]=p+1;
								beta[solver.nCP-cp_models+p]=0.0;
							}
					
							LIBBMRM_MEMCPY(beta_start, beta, solver.nCP*sizeof(float64_t));
							LIBBMRM_MEMCPY(I_start, I, solver.nCP*sizeof(uint32_t));
							qp_cnt=0;
					
							if (tuneAlpha)
							{
								alpha_start=alpha; alpha=0.0;
								LIBBMRM_MEMCPY(I2, I_start, solver.nCP*sizeof(uint32_t));
					
								/* add alpha-dependent terms to H, diag_h and b */
								cp_ptr=CPList_head;
					
								for (uint32_t i=0; i<solver.nCP; ++i)
								{
									A_1=get_cutting_plane(cp_ptr);
									cp_ptr=cp_ptr->next;
					
									rsum = SGVector<float64_t>::dot(A_1, prevW, nDim);
					
									b2[i]=b[i]-((2*alpha)/(_lambda+2*alpha))*rsum;
									diag_H2[i]=diag_H[i]/(_lambda+2*alpha);
					
									for (uint32_t j=0; j<solver.nCP; ++j)
										H2[LIBBMRM_INDEX(i, j, BufSize)]=
											H[LIBBMRM_INDEX(i, j, BufSize)]/(_lambda+2*alpha);
					
								}
					
								/* solve QP with current alpha */
								qp_exitflag=libqp_splx_solver(&get_col, diag_H2, b2, C, I2, S, beta,
										solver.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
								solver.qp_exitflag=qp_exitflag.exitflag;
								qp_cnt++;
								Fd_alpha0=-qp_exitflag.QP;
					
								/* obtain w_t and check if norm(w_{t+1} -w_t) <= K */
								memset(wt, 0, sizeof(float64_t)*nDim);
								SGVector<float64_t>::vec1_plus_scalar_times_vec2(wt, 2*alpha/(_lambda+2*alpha), prevW, nDim);
								cp_ptr=CPList_head;
								for (uint32_t j=0; j<solver.nCP; ++j)
								{
									A_1=get_cutting_plane(cp_ptr);
									cp_ptr=cp_ptr->next;
									SGVector<float64_t>::vec1_plus_scalar_times_vec2(wt, -beta[j]/(_lambda+2*alpha), A_1, nDim);
								}
					
								sq_norm_Wdiff=0.0;
					
								for (uint32_t i=0; i<nDim; ++i)
									sq_norm_Wdiff+=(wt[i]-prevW[i])*(wt[i]-prevW[i]);
					
								if (CMath::sqrt(sq_norm_Wdiff) <= K)
								{
									flag=false;
					
									if (alpha!=alpha_start)
										alphaChanged=true;
								}
								else
								{
									alpha=alpha_start;
								}
					
								while(flag)
								{
									LIBBMRM_MEMCPY(I2, I_start, solver.nCP*sizeof(uint32_t));
									LIBBMRM_MEMCPY(beta, beta_start, solver.nCP*sizeof(float64_t));
					
									/* add alpha-dependent terms to H, diag_h and b */
									cp_ptr=CPList_head;
					
									for (uint32_t i=0; i<solver.nCP; ++i)
									{
										A_1=get_cutting_plane(cp_ptr);
										cp_ptr=cp_ptr->next;
					
										rsum = SGVector<float64_t>::dot(A_1, prevW, nDim);
					
										b2[i]=b[i]-((2*alpha)/(_lambda+2*alpha))*rsum;
										diag_H2[i]=diag_H[i]/(_lambda+2*alpha);
					
										for (uint32_t j=0; j<solver.nCP; ++j)
											H2[LIBBMRM_INDEX(i, j, BufSize)]=H[LIBBMRM_INDEX(i, j, BufSize)]/(_lambda+2*alpha);
									}
					
									/* solve QP with current alpha */
									qp_exitflag=libqp_splx_solver(&get_col, diag_H2, b2, C, I2, S, beta,
											solver.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
									solver.qp_exitflag=qp_exitflag.exitflag;
									qp_cnt++;
					
									/* obtain w_t and check if norm(w_{t+1}-w_t) <= K */
									memset(wt, 0, sizeof(float64_t)*nDim);
									SGVector<float64_t>::vec1_plus_scalar_times_vec2(wt, 2*alpha/(_lambda+2*alpha), prevW, nDim);
									cp_ptr=CPList_head;
									for (uint32_t j=0; j<solver.nCP; ++j)
									{
										A_1=get_cutting_plane(cp_ptr);
										cp_ptr=cp_ptr->next;
										SGVector<float64_t>::vec1_plus_scalar_times_vec2(wt, -beta[j]/(_lambda+2*alpha), A_1, nDim);
									}
					
									sq_norm_Wdiff=0.0;
					
									for (uint32_t i=0; i<nDim; ++i)
										sq_norm_Wdiff+=(wt[i]-prevW[i])*(wt[i]-prevW[i]);
					
									if (CMath::sqrt(sq_norm_Wdiff) > K)
									{
										/* if there is a record of some good solution (i.e. adjust alpha by division by 2) */
					
										if (isThereGoodSolution)
										{
											LIBBMRM_MEMCPY(beta, beta_good, solver.nCP*sizeof(float64_t));
											LIBBMRM_MEMCPY(I2, I_good, solver.nCP*sizeof(uint32_t));
											alpha=alpha_good;
											qp_exitflag=qp_exitflag_good;
											flag=false;
										}
										else
										{
											if (alpha == 0)
											{
												alpha=1.0;
												alphaChanged=true;
											}
											else
											{
												alpha*=2;
												alphaChanged=true;
											}
										}
									}
									else
									{
										if (alpha > 0)
										{
											/* keep good solution and try for alpha /= 2 if previous alpha was 1 */
											LIBBMRM_MEMCPY(beta_good, beta, solver.nCP*sizeof(float64_t));
											LIBBMRM_MEMCPY(I_good, I2, solver.nCP*sizeof(uint32_t));
											alpha_good=alpha;
											qp_exitflag_good=qp_exitflag;
											isThereGoodSolution=true;
					
											if (alpha!=1.0)
											{
												alpha/=2.0;
												alphaChanged=true;
											}
											else
											{
												alpha=0.0;
												alphaChanged=true;
											}
										}
										else
										{
											flag=false;
										}
									}
								}
							}
							else
							{
								alphaChanged=false;
								LIBBMRM_MEMCPY(I2, I_start, solver.nCP*sizeof(uint32_t));
								LIBBMRM_MEMCPY(beta, beta_start, solver.nCP*sizeof(float64_t));
					
								/* add alpha-dependent terms to H, diag_h and b */
								cp_ptr=CPList_head;
					
								for (uint32_t i=0; i<solver.nCP; ++i)
								{
									A_1=get_cutting_plane(cp_ptr);
									cp_ptr=cp_ptr->next;
					
									rsum = SGVector<float64_t>::dot(A_1, prevW, nDim);
					
									b2[i]=b[i]-((2*alpha)/(_lambda+2*alpha))*rsum;
									diag_H2[i]=diag_H[i]/(_lambda+2*alpha);
					
									for (uint32_t j=0; j<solver.nCP; ++j)
										H2[LIBBMRM_INDEX(i, j, BufSize)]=H[LIBBMRM_INDEX(i, j, BufSize)]/(_lambda+2*alpha);
								}
					
								/* solve QP with current alpha */
								qp_exitflag=libqp_splx_solver(&get_col, diag_H2, b2, C, I2, S, beta,
										solver.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
								solver.qp_exitflag=qp_exitflag.exitflag;
								qp_cnt++;
							}
							/* ----------------------------------------------------------------------------------------------- */
					
							/* Update ICPcounter (add one to unused and reset used) + compute number of active CPs */
							solver.nzA=0;
					
							for (uint32_t aaa=0; aaa<solver.nCP; ++aaa)
							{
								if (beta[aaa]>epsilon)
								{
									++solver.nzA;
									icp_stats.ICPcounter[aaa]=0;
								}
								else
								{
									icp_stats.ICPcounter[aaa]+=1;
								}
							}
					
							/* W update */
							memset(W, 0, sizeof(float64_t)*nDim);
							SGVector<float64_t>::vec1_plus_scalar_times_vec2(W, 2*alpha/(_lambda+2*alpha), prevW, nDim);
							cp_ptr=CPList_head;
							for (uint32_t j=0; j<solver.nCP; ++j)
							{
								A_1=get_cutting_plane(cp_ptr);
								cp_ptr=cp_ptr->next;
								SGVector<float64_t>::vec1_plus_scalar_times_vec2(W, -beta[j]/(_lambda+2*alpha), A_1, nDim);
							}
					
							/* risk and subgradient computation */
							R=0.0;
					
							for (uint32_t p=0; p<cp_models; ++p)
							{
								Rt[p] = machine->risk(subgrad_t[p], W, info[p]);
								b[solver.nCP+p] = SGVector<float64_t>::dot(subgrad_t[p], W, nDim) - Rt[p];
								add_cutting_plane(&CPList_tail, map, A, find_free_idx(map, BufSize), subgrad_t[p], nDim);
								R+=Rt[p];
							}
					
							sq_norm_W=SGVector<float64_t>::dot(W, W, nDim);
							sq_norm_prevW=SGVector<float64_t>::dot(prevW, prevW, nDim);
							sq_norm_Wdiff=0.0;
					
							for (uint32_t j=0; j<nDim; ++j)
							{
								sq_norm_Wdiff+=(W[j]-prevW[j])*(W[j]-prevW[j]);
							}
					
							/* compute Fp and Fd */
							solver.Fp=R+0.5*_lambda*sq_norm_W + alpha*sq_norm_Wdiff;
							solver.Fd=-qp_exitflag.QP + ((alpha*_lambda)/(_lambda + 2*alpha))*sq_norm_prevW;
					
							/* gamma + tuneAlpha flag */
							if (alphaChanged)
							{
								eps=1.0-(solver.Fd/solver.Fp);
								gamma=(lastFp*(1-eps)-Fd_alpha0)/(Tmax*(1-eps));
							}
					
							if ((lastFp-solver.Fp) <= gamma)
							{
								tuneAlpha=true;
							}
							else
							{
								tuneAlpha=false;
							}
					
							/* Stopping conditions - set only with nonzero alpha */
							if (alpha==0.0)
							{
								if (solver.Fp-solver.Fd<=TolRel*LIBBMRM_ABS(solver.Fp))
									solver.exitflag=1;
					
								if (solver.Fp-solver.Fd<=TolAbs)
									solver.exitflag=2;
							}
					
							if (solver.nCP>=BufSize)
								solver.exitflag=-1;
					
							tstop=ttime.cur_time_diff(false);
					
							/* compute wdist (= || W_{t+1} - W_{t} || ) */
							sq_norm_Wdiff=0.0;
					
							for (uint32_t i=0; i<nDim; ++i)
							{
								sq_norm_Wdiff+=(W[i]-prevW[i])*(W[i]-prevW[i]);
							}
					
							wdist=CMath::sqrt(sq_norm_Wdiff);
					
							/* Keep history of Fp, Fd and wdist */
							solver.hist_Fp[solver.nIter]=solver.Fp;
							solver.hist_Fd[solver.nIter]=solver.Fd;
							solver.hist_wdist[solver.nIter]=wdist;
					
							/* Verbose output */
							if (verbose)
								SG_SDEBUG("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, nCP=%d, nzA=%d, wdist=%lf, alpha=%lf, qp_cnt=%d, gamma=%lf, tuneAlpha=%d\n",
										solver.nIter, tstop-tstart, solver.Fp, solver.Fd, solver.Fp-solver.Fd,
										(solver.Fp-solver.Fd)/solver.Fp, R, solver.nCP, solver.nzA, wdist, alpha,
										qp_cnt, gamma, tuneAlpha);
					
							/* Check size of Buffer */
							if (solver.nCP>=BufSize)
							{
								solver.exitflag=-2;
								SG_SERROR("Buffer exceeded.\n")
							}
					
							/* keep w_t + Fp */
							LIBBMRM_MEMCPY(prevW, W, nDim*sizeof(float64_t));
							lastFp=solver.Fp;
					
							/* Inactive Cutting Planes (ICP) removal */
							if (cleanICP)
							{
								clean_icp(&icp_stats, p3bmrm, &CPList_head,
										&CPList_tail, H, diag_H, beta, map,
										cleanAfter, b, I, cp_models);
							}
					
							// next CP would exceed BufSize
							if (solver.nCP+1 >= BufSize)
								solver.exitflag=-1;
					
							/* Debug: compute objective and training error */
							if (verbose)
							{
								SGVector<float64_t> w_debug(W, nDim, false);
								float64_t primal = CSOSVMHelper::primal_objective(w_debug, model, _lambda);
								float64_t train_error = CSOSVMHelper::average_loss(w_debug, model);
								helper->add_debug_info(primal, solver.nIter, train_error);
							}
						} /* end of main loop */
					
						if (verbose)
						{
							helper->terminate();
							SG_UNREF(helper);
						}
					
						solver.hist_Fp.resize_vector(solver.nIter);
						solver.hist_Fd.resize_vector(solver.nIter);
						solver.hist_wdist.resize_vector(solver.nIter);
					
						cp_ptr=CPList_head;
					
						while(cp_ptr!=NULL)
						{
							cp_ptr2=cp_ptr;
							cp_ptr=cp_ptr->next;
							LIBBMRM_FREE(cp_ptr2);
							cp_ptr2=NULL;
						}
						cp_list=NULL;
						exitpath-P3BM: 
							LIBBMRM_FREE(subgrad_t);
							LIBBMRM_FREE(Rt);
							LIBBMRM_FREE(wt);
							LIBBMRM_FREE(C);
							LIBBMRM_FREE(S);
							LIBBMRM_FREE(info);
							LIBBMRM_FREE(beta_start);
							LIBBMRM_FREE(beta_good);
							LIBBMRM_FREE(b2);
							LIBBMRM_FREE(diag_H2);
							LIBBMRM_FREE(H2);
							LIBBMRM_FREE(I_start);
							LIBBMRM_FREE(I_good);
							LIBBMRM_FREE(I2);
							goto exitpath;
							
						break;
		}
		
		case PPBMRM : {
					subgrad= (float64_t*) LIBBMRM_CALLOC(nDim, float64_t);
					wt= (float64_t*) LIBBMRM_CALLOC(nDim, float64_t);
					/* Temporary buffers */
					beta_start= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
					beta_good= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
					b2= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
					diag_H2= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);
					H2= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, float64_t);
					I_start= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
					I_good= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
					I2= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
					if (beta_start==NULL || beta_good==NULL || b2==NULL || diag_H2==NULL ||
							I_start==NULL || I_good==NULL || I2==NULL || H2==NULL || wt==NULL
							|| subgrad==NULL)
					{
						solver.exitflag=-2;
						goto exitflag-PPBM;
					}			
					
					solver.hist_Fp.resize_vector(BufSize);
					solver.hist_Fd.resize_vector(BufSize);
					solver.hist_wdist.resize_vector(BufSize);
					
					/* Iinitial solution */
					R = machine->risk(subgrad, W);
					
					solver.nCP=0;
					solver.nIter=0;
					solver.exitflag=0;
					
					b[0]=-R;
					
					/* Cutting plane auxiliary double linked list */
					LIBBMRM_MEMCPY(A, subgrad, nDim*sizeof(float64_t));
					map[0]=false;
					cp_list->address=&A[0];
					cp_list->idx=0;
					cp_list->prev=NULL;
					cp_list->next=NULL;
					CPList_head=cp_list;
					CPList_tail=cp_list;
					
					/* Compute initial value of Fp, Fd, assuming that W is zero vector */
					sq_norm_Wdiff=0.0;
					
					b[0] = SGVector<float64_t>::dot(subgrad, W, nDim);
					sq_norm_W = SGVector<float64_t>::dot(W, W, nDim);
					for (uint32_t j=0; j<nDim; ++j)
					{
						sq_norm_Wdiff+=(W[j]-prevW[j])*(W[j]-prevW[j]);
					}
					
					solver.Fp=R+0.5*_lambda*sq_norm_W + alpha*sq_norm_Wdiff;
					solver.Fd=-LIBBMRM_PLUS_INF;
					lastFp=solver.Fp;
					wdist=CMath::sqrt(sq_norm_Wdiff);
					
					K = (sq_norm_W == 0.0) ? 0.4 : 0.01*CMath::sqrt(sq_norm_W);
					
					LIBBMRM_MEMCPY(prevW, W, nDim*sizeof(float64_t));
					
					tstop=ttime.cur_time_diff(false);
					
					/* Keep history of Fp, Fd, wdist */
					solver.hist_Fp[0]=solver.Fp;
					solver.hist_Fd[0]=solver.Fd;
					solver.hist_wdist[0]=wdist;
					
					/* Verbose output */
					
					if (verbose)
						SG_SDEBUG("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, R=%lf, K=%lf\n",
								solver.nIter, tstop-tstart, solver.Fp, solver.Fd, R, K);
					
					if (verbose)
						helper = machine->get_helper();
					
					/* main loop */
					
					while (solver.exitflag==0)
					{
						tstart=ttime.cur_time_diff(false);
						solver.nIter++;
					
						/* Update H */
					
						if (solver.nCP>0)
						{
							A_2=get_cutting_plane(CPList_tail);
							cp_ptr=CPList_head;
					
							for (uint32_t i=0; i<solver.nCP; ++i)
							{
								A_1=get_cutting_plane(cp_ptr);
								cp_ptr=cp_ptr->next;
								rsum=SGVector<float64_t>::dot(A_1, A_2, nDim);
					
								H[LIBBMRM_INDEX(solver.nCP, i, BufSize)]
									= H[LIBBMRM_INDEX(i, solver.nCP, BufSize)]
									= rsum;
							}
						}
					
						A_2=get_cutting_plane(CPList_tail);
						rsum = SGVector<float64_t>::dot(A_2, A_2, nDim);
					
						H[LIBBMRM_INDEX(solver.nCP, solver.nCP, BufSize)]=rsum;
					
						diag_H[solver.nCP]=H[LIBBMRM_INDEX(solver.nCP, solver.nCP, BufSize)];
						I[solver.nCP]=1;
					
						beta[solver.nCP]=0.0; // [beta; 0]
						solver.nCP++;
					
						/* tune alpha cycle */
						/* ---------------------------------------------------------------------- */
					
						flag=true;
						isThereGoodSolution=false;
						LIBBMRM_MEMCPY(beta_start, beta, solver.nCP*sizeof(float64_t));
						LIBBMRM_MEMCPY(I_start, I, solver.nCP*sizeof(uint32_t));
						qp_cnt=0;
						alpha_good=alpha;
					
						if (tuneAlpha)
						{
							alpha_start=alpha; alpha=0.0;
							beta[solver.nCP]=0.0;
							LIBBMRM_MEMCPY(I2, I_start, solver.nCP*sizeof(uint32_t));
							I2[solver.nCP]=1;
					
							/* add alpha-dependent terms to H, diag_h and b */
							cp_ptr=CPList_head;
					
							for (uint32_t i=0; i<solver.nCP; ++i)
							{
								A_1=get_cutting_plane(cp_ptr);
								cp_ptr=cp_ptr->next;
					
								rsum = SGVector<float64_t>::dot(A_1, prevW, nDim);
					
								b2[i]=b[i]-((2*alpha)/(_lambda+2*alpha))*rsum;
								diag_H2[i]=diag_H[i]/(_lambda+2*alpha);
					
								for (uint32_t j=0; j<solver.nCP; ++j)
									H2[LIBBMRM_INDEX(i, j, BufSize)]=
										H[LIBBMRM_INDEX(i, j, BufSize)]/(_lambda+2*alpha);
							}
					
							/* solve QP with current alpha */
							qp_exitflag=libqp_splx_solver(&get_col, diag_H2, b2, &Cx, I2, &Sx, beta,
									solver.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
							solver.qp_exitflag=qp_exitflag.exitflag;
							qp_cnt++;
							Fd_alpha0=-qp_exitflag.QP;
					
							/* obtain w_t and check if norm(w_{t+1} -w_t) <= K */
							for (uint32_t i=0; i<nDim; ++i)
							{
								rsum=0.0;
								cp_ptr=CPList_head;
					
								for (uint32_t j=0; j<solver.nCP; ++j)
								{
									A_1=get_cutting_plane(cp_ptr);
									cp_ptr=cp_ptr->next;
									rsum+=A_1[i]*beta[j];
								}
					
								wt[i]=(2*alpha*prevW[i] - rsum)/(_lambda+2*alpha);
							}
					
							sq_norm_Wdiff=0.0;
					
							for (uint32_t i=0; i<nDim; ++i)
								sq_norm_Wdiff+=(wt[i]-prevW[i])*(wt[i]-prevW[i]);
					
							if (CMath::sqrt(sq_norm_Wdiff) <= K)
							{
								flag=false;
					
								if (alpha!=alpha_start)
									alphaChanged=true;
							}
							else
							{
								alpha=alpha_start;
							}
					
							while(flag)
							{
								LIBBMRM_MEMCPY(I2, I_start, solver.nCP*sizeof(uint32_t));
								LIBBMRM_MEMCPY(beta, beta_start, solver.nCP*sizeof(float64_t));
								I2[solver.nCP]=1;
								beta[solver.nCP]=0.0;
					
								/* add alpha-dependent terms to H, diag_h and b */
								cp_ptr=CPList_head;
					
								for (uint32_t i=0; i<solver.nCP; ++i)
								{
									A_1=get_cutting_plane(cp_ptr);
									cp_ptr=cp_ptr->next;
					
									rsum = SGVector<float64_t>::dot(A_1, prevW, nDim);
					
									b2[i]=b[i]-((2*alpha)/(_lambda+2*alpha))*rsum;
									diag_H2[i]=diag_H[i]/(_lambda+2*alpha);
					
									for (uint32_t j=0; j<solver.nCP; ++j)
										H2[LIBBMRM_INDEX(i, j, BufSize)]=H[LIBBMRM_INDEX(i, j, BufSize)]/(_lambda+2*alpha);
								}
					
								/* solve QP with current alpha */
								qp_exitflag=libqp_splx_solver(&get_col, diag_H2, b2, &Cx, I2, &Sx, beta,
										solver.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
								solver.qp_exitflag=qp_exitflag.exitflag;
								qp_cnt++;
					
								/* obtain w_t and check if norm(w_{t+1}-w_t) <= K */
								for (uint32_t i=0; i<nDim; ++i)
								{
									rsum=0.0;
									cp_ptr=CPList_head;
					
									for (uint32_t j=0; j<solver.nCP; ++j)
									{
										A_1=get_cutting_plane(cp_ptr);
										cp_ptr=cp_ptr->next;
										rsum+=A_1[i]*beta[j];
									}
					
									wt[i]=(2*alpha*prevW[i] - rsum)/(_lambda+2*alpha);
								}
					
								sq_norm_Wdiff=0.0;
								for (uint32_t i=0; i<nDim; ++i)
									sq_norm_Wdiff+=(wt[i]-prevW[i])*(wt[i]-prevW[i]);
					
								if (CMath::sqrt(sq_norm_Wdiff) > K)
								{
									/* if there is a record of some good solution
									 * (i.e. adjust alpha by division by 2) */
					
									if (isThereGoodSolution)
									{
										LIBBMRM_MEMCPY(beta, beta_good, solver.nCP*sizeof(float64_t));
										LIBBMRM_MEMCPY(I2, I_good, solver.nCP*sizeof(uint32_t));
										alpha=alpha_good;
										qp_exitflag=qp_exitflag_good;
										flag=false;
									}
									else
									{
										if (alpha == 0)
										{
											alpha=1.0;
											alphaChanged=true;
										}
										else
										{
											alpha*=2;
											alphaChanged=true;
										}
									}
								}
								else
								{
									if (alpha > 0)
									{
										/* keep good solution and try for alpha /= 2 if previous alpha was 1 */
										LIBBMRM_MEMCPY(beta_good, beta, solver.nCP*sizeof(float64_t));
										LIBBMRM_MEMCPY(I_good, I2, solver.nCP*sizeof(uint32_t));
										alpha_good=alpha;
										qp_exitflag_good=qp_exitflag;
										isThereGoodSolution=true;
					
										if (alpha!=1.0)
										{
											alpha/=2.0;
											alphaChanged=true;
										}
										else
										{
											alpha=0.0;
											alphaChanged=true;
										}
									}
									else
									{
										flag=false;
									}
								}
							}
						}
						else
						{
							alphaChanged=false;
							LIBBMRM_MEMCPY(I2, I_start, solver.nCP*sizeof(uint32_t));
							LIBBMRM_MEMCPY(beta, beta_start, solver.nCP*sizeof(float64_t));
					
							/* add alpha-dependent terms to H, diag_h and b */
							cp_ptr=CPList_head;
					
							for (uint32_t i=0; i<solver.nCP; ++i)
							{
								A_1=get_cutting_plane(cp_ptr);
								cp_ptr=cp_ptr->next;
					
								rsum = SGVector<float64_t>::dot(A_1, prevW, nDim);
					
								b2[i]=b[i]-((2*alpha)/(_lambda+2*alpha))*rsum;
								diag_H2[i]=diag_H[i]/(_lambda+2*alpha);
					
								for (uint32_t j=0; j<solver.nCP; ++j)
									H2[LIBBMRM_INDEX(i, j, BufSize)]=
										H[LIBBMRM_INDEX(i, j, BufSize)]/(_lambda+2*alpha);
							}
							/* solve QP with current alpha */
							qp_exitflag=libqp_splx_solver(&get_col, diag_H2, b2, &Cx, I2, &Sx, beta,
									solver.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
							solver.qp_exitflag=qp_exitflag.exitflag;
							qp_cnt++;
						}
					
						/* ----------------------------------------------------------------------------------------------- */
					
						/* Update ICPcounter (add one to unused and reset used) + compute number of active CPs */
						solver.nzA=0;
					
						for (uint32_t aaa=0; aaa<solver.nCP; ++aaa)
						{
							if (beta[aaa]>epsilon)
							{
								++solver.nzA;
								icp_stats.ICPcounter[aaa]=0;
							}
							else
							{
								icp_stats.ICPcounter[aaa]+=1;
							}
						}
					
						/* W update */
						for (uint32_t i=0; i<nDim; ++i)
						{
							rsum=0.0;
							cp_ptr=CPList_head;
					
							for (uint32_t j=0; j<solver.nCP; ++j)
							{
								A_1=get_cutting_plane(cp_ptr);
								cp_ptr=cp_ptr->next;
								rsum+=A_1[i]*beta[j];
							}
					
							W[i]=(2*alpha*prevW[i]-rsum)/(_lambda+2*alpha);
						}
					
						/* risk and subgradient computation */
						R = machine->risk(subgrad, W);
						add_cutting_plane(&CPList_tail, map, A,
								find_free_idx(map, BufSize), subgrad, nDim);
					
						sq_norm_W=SGVector<float64_t>::dot(W, W, nDim);
						sq_norm_prevW=SGVector<float64_t>::dot(prevW, prevW, nDim);
						b[solver.nCP]=SGVector<float64_t>::dot(subgrad, W, nDim) - R;
					
						sq_norm_Wdiff=0.0;
						for (uint32_t j=0; j<nDim; ++j)
						{
							sq_norm_Wdiff+=(W[j]-prevW[j])*(W[j]-prevW[j]);
						}
					
						/* compute Fp and Fd */
						solver.Fp=R+0.5*_lambda*sq_norm_W + alpha*sq_norm_Wdiff;
						solver.Fd=-qp_exitflag.QP+((alpha*_lambda)/(_lambda + 2*alpha))*sq_norm_prevW;
					
						/* gamma + tuneAlpha flag */
						if (alphaChanged)
						{
							eps=1.0-(solver.Fd/solver.Fp);
							gamma=(lastFp*(1-eps)-Fd_alpha0)/(Tmax*(1-eps));
						}
					
						if ((lastFp-solver.Fp) <= gamma)
						{
							tuneAlpha=true;
						}
						else
						{
							tuneAlpha=false;
						}
					
						/* Stopping conditions - set only with nonzero alpha */
						if (alpha==0.0)
						{
							if (solver.Fp-solver.Fd<=TolRel*LIBBMRM_ABS(solver.Fp))
								solver.exitflag=1;
					
							if (solver.Fp-solver.Fd<=TolAbs)
								solver.exitflag=2;
						}
					
						if (solver.nCP>=BufSize)
							solver.exitflag=-1;
					
						tstop=ttime.cur_time_diff(false);
					
						/* compute wdist (= || W_{t+1} - W_{t} || ) */
						sq_norm_Wdiff=0.0;
					
						for (uint32_t i=0; i<nDim; ++i)
						{
							sq_norm_Wdiff+=(W[i]-prevW[i])*(W[i]-prevW[i]);
						}
					
						wdist=CMath::sqrt(sq_norm_Wdiff);
					
						/* Keep history of Fp, Fd, wdist */
						solver.hist_Fp[solver.nIter]=solver.Fp;
						solver.hist_Fd[solver.nIter]=solver.Fd;
						solver.hist_wdist[solver.nIter]=wdist;
					
						/* Verbose output */
						if (verbose)
							SG_SDEBUG("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, nCP=%d, nzA=%d, wdist=%lf, alpha=%lf, qp_cnt=%d, gamma=%lf, tuneAlpha=%d\n",
									solver.nIter, tstop-tstart, solver.Fp, solver.Fd, solver.Fp-solver.Fd,
									(solver.Fp-solver.Fd)/solver.Fp, R, solver.nCP, solver.nzA, wdist, alpha,
									qp_cnt, gamma, tuneAlpha);
					
						/* Check size of Buffer */
						if (solver.nCP>=BufSize)
						{
							solver.exitflag=-2;
							SG_SERROR("Buffer exceeded.\n")
						}
					
						/* keep w_t + Fp */
						LIBBMRM_MEMCPY(prevW, W, nDim*sizeof(float64_t));
						lastFp=solver.Fp;
					
						/* Inactive Cutting Planes (ICP) removal */
						if (cleanICP)
						{
							clean_icp(&icp_stats, ppbmrm, &CPList_head, &CPList_tail, H, diag_H, beta, map, cleanAfter, b, I);
						}
					
						// next CP would exceed BufSize
						if (solver.nCP+1 >= BufSize)
							solver.exitflag=-1;
					
						/* Debug: compute objective and training error */
						if (verbose)
						{
							SGVector<float64_t> w_debug(W, nDim, false);
							float64_t primal = CSOSVMHelper::primal_objective(w_debug, model, _lambda);
							float64_t train_error = CSOSVMHelper::average_loss(w_debug, model);
							helper->add_debug_info(primal, solver.nIter, train_error);
						}
					} /* end of main loop */
					
					if (verbose)
					{
						helper->terminate();
						SG_UNREF(helper);
					}
					
					solver.hist_Fp.resize_vector(solver.nIter);
					solver.hist_Fd.resize_vector(solver.nIter);
					solver.hist_wdist.resize_vector(solver.nIter);
					
					cp_ptr=CPList_head;
					
					while(cp_ptr!=NULL)
					{
						cp_ptr2=cp_ptr;
						cp_ptr=cp_ptr->next;
						LIBBMRM_FREE(cp_ptr2);
						cp_ptr2=NULL;
					}
					
					cp_list=NULL;
					exitpath-PPBM:
						LIBBMRM_FREE(beta_start);
						LIBBMRM_FREE(beta_good);
						LIBBMRM_FREE(b2);
						LIBBMRM_FREE(diag_H2);
						LIBBMRM_FREE(I_start);
						LIBBMRM_FREE(I_good);
						LIBBMRM_FREE(I2);
						LIBBMRM_FREE(H2);
						LIBBMRM_FREE(wt);
						LIBBMRM_FREE(subgrad);
						goto exitpath;
						
					break;
		}
		default: { /*The Solver is not present*/
					goto exitpath;
					break;
		}
		
		
	}
	
	exitpath:
		LIBBMRM_FREE(H);
		LIBBMRM_FREE(A);
		LIBBMRM_FREE(b);
		LIBBMRM_FREE(beta);
		LIBBMRM_FREE(diag);
		LIBBMRM_FREE(diag_H);
		LIBBMRM_FREE(I);
		LIBBMRM_FREE(cp_list);
		LIBBMRM_FREE(prevW);
		LIBBMRM_FREE(icp_stats.ICPcounter);
		LIBBMRM_FREE(icp_stats.ICPs);
		LIBBMRM_FREE(icp_stats.ACPs);
		LIBBMRM_FREE(icp_stats.H_buff);

	if (cp_list)
		LIBBMRM_FREE(cp_list);

	SG_UNREF(model);
	
	return (solver);
}
}
