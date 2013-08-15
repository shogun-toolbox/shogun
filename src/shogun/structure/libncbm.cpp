/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Viktor Gal
 *
 */

#include <shogun/structure/libncbm.h>
#include <shogun/lib/Time.h>

#include <shogun/lib/external/libqp.h>
#include <shogun/multiclass/GMNPLib.h>

#include <vector>

namespace shogun
{

static float64_t* HMatrix;
static uint32_t maxCPs;
static const float64_t epsilon=0.0;

static const float64_t *get_col(uint32_t i)
{
	return (&HMatrix[maxCPs*i]);
}

IGNORE_IN_CLASSLIST struct line_search_res
{
	/* */
	float64_t a;
	float64_t fval;
	float64_t reg;
	SGVector<float64_t> solution;
	SGVector<float64_t> gradient;
};

inline static line_search_res zoom
(
 CDualLibQPBMSOSVM *machine, 
 float64_t lambda,
 float64_t a_lo,
 float64_t a_hi,
 float64_t initial_fval,
 SGVector<float64_t>& initial_solution,
 SGVector<float64_t>& search_dir,
 float64_t wolfe_c1,
 float64_t wolfe_c2,
 float64_t init_lgrad,
 float64_t f_lo,
 float64_t g_lo,
 float64_t f_hi,
 float64_t g_hi
)
{
	line_search_res ls_res;
	ls_res.solution.resize_vector(initial_solution.vlen);
	ls_res.gradient.resize_vector(initial_solution.vlen);
	
	SGVector<float64_t> cur_solution(initial_solution.vlen);
	cur_solution.zero();
	SGVector<float64_t> cur_grad(initial_solution.vlen);

	uint32_t iter = 0;
	while (1)
	{
		float64_t d1 = g_lo+g_hi - 3*(f_lo-f_hi)/(a_lo-a_hi);
		float64_t d2 = CMath::sqrt(d1*d1 - g_lo*g_hi);
		float64_t a_j = a_hi -(a_hi-a_lo)*(g_hi+d2-d1)/(g_hi-g_lo+2*d2);
		
		if (a_lo < a_hi)
		{
			if ((a_j < a_lo) || (a_j > a_hi))
			{
				a_j = 0.5*(a_lo+a_hi);
			}
		}
		else
		{
			if ((a_j > a_lo) || (a_j < a_hi))
			{
				a_j = 0.5*(a_lo+a_hi);
			}
		}
		
		cur_solution.add(cur_solution.vector, 1.0,
				initial_solution.vector, a_j,
				search_dir.vector, cur_solution.vlen);

		float64_t cur_fval = machine->risk(cur_grad.vector, cur_solution.vector);
		float64_t cur_reg
			= 0.5*lambda*cur_solution.dot(cur_solution.vector,
					cur_solution.vector, cur_solution.vlen);
		cur_fval += cur_reg;

		cur_grad.vec1_plus_scalar_times_vec2(cur_grad.vector, lambda, cur_solution.vector, cur_grad.vlen);

		if
			(
			 (cur_fval > (initial_fval+wolfe_c1*a_j*init_lgrad))
			 ||
			 (cur_fval > f_lo)
			)
		{
			a_hi = a_j;
			f_hi = cur_fval;
			g_hi = cur_grad.dot(cur_grad.vector, search_dir.vector, cur_grad.vlen);
		}
		else
		{
			float64_t cur_lgrad
				= cur_grad.dot(cur_grad.vector, search_dir.vector, cur_grad.vlen);

			if (CMath::abs(cur_lgrad) < -wolfe_c2*init_lgrad)
			{
				ls_res.a = a_j;
				ls_res.fval = cur_fval;
				ls_res.reg = cur_reg;
				ls_res.gradient = cur_grad;
				ls_res.solution = cur_solution;
//				SG_SPRINT("in zoom (wolfe2): %f\n", cur_fval)
				return ls_res;
			}

			if (cur_lgrad*(a_hi - a_lo) >= 0)
			{
				a_hi = a_lo;
				f_hi = f_lo;
				g_hi = g_lo;
			}
			a_lo = a_j;
			f_lo = cur_fval;
			g_lo = cur_lgrad;
		}
		
		if
			(
			 (CMath::abs(a_lo - a_hi) <= 0.01*a_lo)
			 ||
			 (iter >= 5)
			)
		{
			ls_res.a = a_j;
			ls_res.fval = cur_fval;
			ls_res.reg = cur_reg;
			ls_res.gradient = cur_grad;
			ls_res.solution = cur_solution;
//			SG_SPRINT("in zoom iter: %d %f\n", iter, cur_fval)
			return ls_res;
		}
		iter++;
	}
}

inline std::vector<line_search_res> line_search_with_strong_wolfe
(
		CDualLibQPBMSOSVM *machine, 
		float64_t lambda,
		float64_t initial_val,
		SGVector<float64_t>& initial_solution,
		SGVector<float64_t>& initial_grad,
		SGVector<float64_t>& search_dir,
		float64_t astart,
		float64_t amax = 1.1,
		float64_t wolfe_c1 = 1E-4,
		float64_t wolfe_c2 = 0.9,
		float64_t max_iter = 5
)
{
	/* NOTE:
	 * model->risk returns only the risk as it's name says
	 * have to cur_fval = model.risk + (lambda*0.5*w*w')
	 *
	 * subgrad should be: subgrad + (lambda*w)
	 *
	 */
	
	uint32_t iter = 0;

	initial_grad.vec1_plus_scalar_times_vec2(initial_grad.vector, lambda, initial_solution.vector, initial_grad.vlen);

	float64_t initial_lgrad = initial_grad.dot(initial_grad.vector, search_dir.vector, initial_grad.vlen);
	float64_t prev_lgrad = initial_lgrad;
	float64_t prev_fval = initial_val;

	float64_t prev_a = 0;
	float64_t cur_a = astart;

	std::vector<line_search_res> ret;
	while (1)
	{
		SGVector<float64_t> x(initial_solution.vlen);
		SGVector<float64_t> cur_subgrad(initial_solution.vlen);

		x.add(x.vector, 1.0, initial_solution.vector, cur_a, search_dir.vector, x.vlen);
		float64_t cur_fval = machine->risk(cur_subgrad.vector, x.vector);
		float64_t cur_reg = 0.5*lambda*x.dot(x.vector, x.vector, x.vlen);
		cur_fval += cur_reg;

		cur_subgrad.vec1_plus_scalar_times_vec2(cur_subgrad.vector, lambda, x.vector, x.vlen);
		if (iter == 0)
		{
			line_search_res initial_step;
			initial_step.fval = cur_fval;
			initial_step.reg = cur_reg;
			initial_step.gradient = cur_subgrad;
			initial_step.solution = x;
			ret.push_back(initial_step);
		}

		float64_t cur_lgrad
			= cur_subgrad.dot(cur_subgrad.vector, search_dir.vector,
					cur_subgrad.vlen);
		if
			(
			 (cur_fval > initial_val+wolfe_c1*cur_a*initial_lgrad)
			 ||
			 (cur_fval >= prev_fval && iter > 0)
			)
		{
			ret.push_back(
					zoom(machine, lambda, prev_a, cur_a, initial_val,
						initial_solution, search_dir, wolfe_c1, wolfe_c2,
						initial_lgrad, prev_fval, prev_lgrad, cur_fval, cur_lgrad)
					);
			return ret;
		}

		if (CMath::abs(cur_lgrad) <= -wolfe_c2*initial_lgrad)
		{
			line_search_res ls_res;
			ls_res.a = cur_a;
			ls_res.fval = cur_fval;
			ls_res.reg = cur_reg;
			ls_res.solution = x;
			ls_res.gradient = cur_subgrad;
			ret.push_back(ls_res);
			return ret;
		}

		if (cur_lgrad >= 0)
		{
			ret.push_back(
					zoom(machine, lambda, cur_a, prev_a, initial_val,
						initial_solution, search_dir, wolfe_c1, wolfe_c2,
						initial_lgrad, cur_fval, cur_lgrad, prev_fval, prev_lgrad)
					);
			return ret;
		}
		iter++;
		if ((CMath::abs(cur_a - amax) <= 0.01*amax) || (iter >= max_iter))
		{
			line_search_res ls_res;
			ls_res.a = cur_a;
			ls_res.fval = cur_fval;
			ls_res.reg = cur_reg;
			ls_res.solution = x;
			ls_res.gradient = cur_subgrad;
			ret.push_back(ls_res);
			return ret;
		}

		prev_a = cur_a;
		prev_fval = cur_fval;
		prev_lgrad = cur_lgrad;

		cur_a = (cur_a + amax)*0.5;
	}
}

inline void update_H(BmrmStatistics& ncbm,
		bmrm_ll* head,
		bmrm_ll* tail,
		SGMatrix<float64_t>& H,
		SGVector<float64_t>& diag_H,
		float64_t lambda,
		uint32_t maxCP,
		int32_t w_dim)
{
	float64_t* a_2 = get_cutting_plane(tail);
	bmrm_ll* cp_ptr=head;

	for (uint32_t i=0; i < ncbm.nCP; ++i)
	{
		float64_t* a_1 = get_cutting_plane(cp_ptr);
		cp_ptr=cp_ptr->next;

		float64_t dot_val = SGVector<float64_t>::dot(a_2, a_1, w_dim);

		H.matrix[LIBBMRM_INDEX(ncbm.nCP, i, maxCP)]
			= H.matrix[LIBBMRM_INDEX(i, ncbm.nCP, maxCP)]
			= dot_val/lambda;
	}

	/* set the diagonal element, i.e. subgrad_i*subgrad_i' */
	float64_t dot_val = SGVector<float64_t>::dot(a_2, a_2, w_dim);
	H[LIBBMRM_INDEX(ncbm.nCP, ncbm.nCP, maxCP)]=dot_val/lambda;

	diag_H[ncbm.nCP]=H[LIBBMRM_INDEX(ncbm.nCP, ncbm.nCP, maxCP)];

	ncbm.nCP++;
}


BmrmStatistics svm_ncbm_solver(
		CDualLibQPBMSOSVM *machine, 
		float64_t         *w,
		float64_t         TolRel,
		float64_t         TolAbs,
		float64_t         _lambda,
		uint32_t          _BufSize,
		bool              cleanICP,
		uint32_t          cleanAfter,
		bool              is_convex,
		bool              line_search,
		bool              verbose
		)
{
	BmrmStatistics ncbm;
	libqp_state_T qp_exitflag={0, 0, 0, 0};
	int32_t w_dim = machine->get_model()->get_dim();

	maxCPs = _BufSize;

	ncbm.nCP=0;
	ncbm.nIter=0;
	ncbm.exitflag=0;

	/* variables for timing the algorithm*/
	CTime ttime;
	float64_t tstart, tstop;
	tstart=ttime.cur_time_diff(false);

	/* matrix of subgradiends */
	SGMatrix<float64_t> A(w_dim, maxCPs);

	/* bias vector */
	SGVector<float64_t> bias(maxCPs);
	bias.zero();

	/* Matrix for storing H = A*A' */
	SGMatrix<float64_t> H(maxCPs,maxCPs);
	HMatrix = H.matrix;

	/* diag_H */
	SGVector<float64_t> diag_H(maxCPs);
	diag_H.zero();

	/* x the solution vector of the dual problem:
	 * 1/lambda x*H*x' - x*bias
	 */
	SGVector<float64_t> x(maxCPs);
	x.zero();

	/* for sum_{i in I_k} x[i] <= b[k] for all k such that S[k] == 1 */
	float64_t b = 1.0;
	uint8_t S = 1;
	SGVector<uint32_t> I(maxCPs);
	I.set_const(1);

	/* libqp paramteres */
	uint32_t QPSolverMaxIter = 0xFFFFFFFF;
	float64_t QPSolverTolRel = 1E-9;

	/* variables for maintaining inactive planes */
	SGVector<bool> map(maxCPs);
	map.set_const(true);
	ICP_stats icp_stats;
	icp_stats.maxCPs = maxCPs;
	icp_stats.ICPcounter = (uint32_t*) LIBBMRM_CALLOC(maxCPs, uint32_t);
	icp_stats.ICPs = (float64_t**) LIBBMRM_CALLOC(maxCPs, float64_t*);
	icp_stats.ACPs = (uint32_t*) LIBBMRM_CALLOC(maxCPs, uint32_t);
	icp_stats.H_buff = (float64_t*) LIBBMRM_CALLOC(maxCPs*maxCPs,float64_t);
	if
	(
		icp_stats.ICPcounter == NULL || icp_stats.ICPs == NULL
		|| icp_stats.ACPs == NULL || icp_stats.H_buff == NULL
	)
	{
		ncbm.exitflag=-2;
		LIBBMRM_FREE(icp_stats.ICPcounter);
		LIBBMRM_FREE(icp_stats.ICPs);
		LIBBMRM_FREE(icp_stats.ACPs);
		LIBBMRM_FREE(icp_stats.H_buff);
		return ncbm;
	}

	/* best */
	float64_t best_Fp = CMath::INFTY;
	float64_t best_risk = CMath::INFTY;
	SGVector<float64_t> best_w(w_dim);
	best_w.zero();
	SGVector<float64_t> best_subgrad(w_dim);
	best_subgrad.zero();

	/* initial solution */
	SGVector<float64_t> cur_subgrad(w_dim);
	SGVector<float64_t> cur_w(w_dim);
	memcpy(cur_w.vector, w, sizeof(float64_t)*w_dim);

	float64_t cur_risk = machine->risk(cur_subgrad.vector, cur_w.vector);
	bias[0] = -cur_risk;
	best_Fp = 0.5*_lambda*cur_w.dot(cur_w.vector, cur_w.vector, cur_w.vlen) + cur_risk;
	best_risk = cur_risk;
	memcpy(best_w.vector, cur_w.vector, sizeof(float64_t)*w_dim);
	memcpy(best_subgrad.vector, cur_subgrad.vector, sizeof(float64_t)*w_dim);

	/* create a double-linked list over the A the subgrad matrix */
	bmrm_ll *CPList_head, *CPList_tail, *cp_ptr, *cp_list=NULL;
	cp_list = (bmrm_ll*) SG_CALLOC(bmrm_ll, 1);
	if (cp_list==NULL)
	{
		ncbm.exitflag=-2;
		return ncbm;
	}
	/* save the subgradient */
	memcpy(A.matrix, cur_subgrad.vector, sizeof(float64_t)*w_dim);
	map[0] = false;
	cp_list->address=&A[0];
	cp_list->idx=0;
	cp_list->prev=NULL;
	cp_list->next=NULL;
	CPList_head=cp_list;
	CPList_tail=cp_list;

	update_H(ncbm, CPList_head, CPList_tail, H, diag_H, _lambda, maxCPs, w_dim);
	tstop=ttime.cur_time_diff(false);
	if (verbose)
		SG_SPRINT("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, R=%lf\n",
				ncbm.nIter, tstop-tstart, ncbm.Fp, ncbm.Fd, cur_risk);

	float64_t astar = 0.01;

	SG_SPRINT("clean icps: %d\n", cleanICP)
	while (ncbm.exitflag==0)
	{
		tstart=ttime.cur_time_diff(false);
		ncbm.nIter++;

		//diag_H.display_vector();
		//bias.display_vector();

		/* solve the dual of the problem, namely:
		 *
		 */
		qp_exitflag =
			libqp_splx_solver(&get_col, diag_H.vector, bias.vector, &b, I.vector, &S, x.vector,
					ncbm.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);

		ncbm.Fd = -qp_exitflag.QP;

		ncbm.qp_exitflag=qp_exitflag.exitflag;

		/* Update ICPcounter (add one to unused and reset used)
		 * + compute number of active CPs */
		ncbm.nzA=0;

		for (uint32_t i=0; i < ncbm.nCP; ++i)
		{
			if (x[i] > epsilon)
			{
				/* cp was active => reset counter */
				++ncbm.nzA;
				icp_stats.ICPcounter[i]=0;
			}
			else
			{
				icp_stats.ICPcounter[i]++;
			}
		}

		/* Inactive Cutting Planes (ICP) removal */
		if (cleanICP)
		{
			clean_icp(&icp_stats, ncbm, &CPList_head, &CPList_tail,
					H.matrix, diag_H.vector, x.vector,
					map.vector, cleanAfter, bias.vector, I.vector);
		}

		/* calculate the new w
		 * w[i] = -1/lambda*A[i]*x[i]
		 */
		cur_w.zero();
		cp_ptr=CPList_head;
		for (uint32_t i=0; i < ncbm.nCP; ++i)
		{
			float64_t* A_1 = get_cutting_plane(cp_ptr);
			cp_ptr=cp_ptr->next;
			SGVector<float64_t>::vec1_plus_scalar_times_vec2(cur_w.vector, -x[i]/_lambda, A_1, w_dim);
		}

		bool calc_gap = false;
		if (calc_gap)
		{
			SGVector<float64_t> scores(ncbm.nCP);
			cp_ptr=CPList_head;

			for (uint32_t i=0; i < ncbm.nCP; ++i)
			{
				float64_t* a_1 = get_cutting_plane(cp_ptr);
				cp_ptr = cp_ptr->next;
				scores[i] = cur_w.dot(cur_w.vector, a_1, w_dim);
			}
			scores.vec1_plus_scalar_times_vec2(scores.vector, -1.0, bias.vector, scores.vlen);

			float64_t w_norm = cur_w.dot(cur_w.vector, cur_w.vector, cur_w.vlen);
			float64_t PO = 0.5*_lambda*w_norm + scores.max(scores.vector, scores.vlen);
			float64_t QP_gap = PO - ncbm.Fd;

			SG_SPRINT("%4d: primal:%f dual:%f QP_gap:%f\n", ncbm.nIter, PO, ncbm.Fd, QP_gap)
		}

		/* Stopping conditions */
		if ((best_Fp - ncbm.Fd) <= TolRel*LIBBMRM_ABS(best_Fp))
			ncbm.exitflag = 1;

		if ((best_Fp - ncbm.Fd) <= TolAbs)
			ncbm.exitflag = 2;

		if (ncbm.nCP >= maxCPs)
			ncbm.exitflag = -1;

		tstop=ttime.cur_time_diff(false);

		/* Verbose output */
		if (verbose)
			SG_SPRINT("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, nCP=%d, nzA=%d, QPexitflag=%d, best_fp=%f, gap=%f\n",
					ncbm.nIter, tstop-tstart, ncbm.Fp, ncbm.Fd, ncbm.Fp-ncbm.Fd,
					(ncbm.Fp-ncbm.Fd)/ncbm.Fp, cur_risk, ncbm.nCP, ncbm.nzA, qp_exitflag.exitflag, best_Fp, (best_Fp-ncbm.Fd)/best_Fp);

		std::vector<line_search_res> wbest_candidates;
		if (!line_search)
		{
			cur_risk = machine->risk(cur_subgrad.vector, cur_w.vector);

			add_cutting_plane(&CPList_tail, map, A.matrix,
					find_free_idx(map, maxCPs), cur_subgrad.vector, w_dim);

			bias[ncbm.nCP] = cur_w.dot(cur_w.vector, cur_subgrad.vector, cur_w.vlen) - cur_risk;

			update_H(ncbm, CPList_head, CPList_tail, H, diag_H, _lambda, maxCPs, w_dim);

			// add as a new wbest candidate
			line_search_res ls;
			ls.fval = cur_risk+0.5*_lambda*cur_w.dot(cur_w.vector, cur_w.vector, cur_w.vlen);
			ls.solution = cur_w;
			ls.gradient = cur_subgrad;

			wbest_candidates.push_back(ls);
		}
		else
		{
			tstart=ttime.cur_time_diff(false);
			/* do line searching */
			SGVector<float64_t> search_dir(w_dim);
			search_dir.add(search_dir.vector, 1.0, cur_w.vector, -1.0, best_w.vector, w_dim);

			float64_t norm_dir = search_dir.twonorm(search_dir.vector, search_dir.vlen);
			float64_t astart;
			uint32_t cp_min_approx = 0;
			if (cp_min_approx || (ncbm.nIter == 1))
			{
				astart = 1.0;
			}
			else
			{
				astart = CMath::min(astar/norm_dir,1.0);
				if (astart == 0)
					astart = 1.0;
			}

			/* line search */
			std::vector<line_search_res> ls_res
				= line_search_with_strong_wolfe(machine, _lambda, best_Fp, best_w, best_subgrad, search_dir, astart);

			if (ls_res[0].fval != ls_res[1].fval)
			{
				ls_res[0].gradient.vec1_plus_scalar_times_vec2(ls_res[0].gradient.vector, -_lambda, ls_res[0].solution.vector, w_dim);

				add_cutting_plane(&CPList_tail, map, A.matrix,
						find_free_idx(map, maxCPs), ls_res[0].gradient, w_dim);

				bias[ncbm.nCP]
					= SGVector<float64_t>::dot(ls_res[0].solution.vector, ls_res[0].gradient, w_dim)
					- (ls_res[0].fval - ls_res[0].reg);

				update_H(ncbm, CPList_head, CPList_tail, H, diag_H, _lambda, maxCPs, w_dim);

				wbest_candidates.push_back(ls_res[0]);
			}

			ls_res[1].gradient.vec1_plus_scalar_times_vec2(ls_res[1].gradient.vector, -_lambda, ls_res[1].solution.vector, w_dim);

			add_cutting_plane(&CPList_tail, map, A.matrix,
					find_free_idx(map, maxCPs), ls_res[1].gradient.vector, w_dim);

			bias[ncbm.nCP]
				= ls_res[1].solution.dot(ls_res[1].solution.vector, ls_res[1].gradient.vector, w_dim)
				- (ls_res[1].fval - ls_res[1].reg);

			update_H(ncbm, CPList_head, CPList_tail, H, diag_H, _lambda, maxCPs, w_dim);

			wbest_candidates.push_back(ls_res[1]);

			if ((best_Fp <= ls_res[1].fval) && (astart != 1))
			{
				cur_risk = machine->risk(cur_subgrad.vector, cur_w.vector);

				add_cutting_plane(&CPList_tail, map, A.matrix,
							find_free_idx(map, maxCPs), cur_subgrad.vector, w_dim);

				bias[ncbm.nCP]
					=  cur_w.dot(cur_w.vector, cur_subgrad.vector, cur_w.vlen) - cur_risk;

				update_H(ncbm, CPList_head, CPList_tail, H, diag_H, _lambda, maxCPs, w_dim);
				
				/* add as a new wbest candidate */
				line_search_res ls;
				ls.fval = cur_risk+0.5*_lambda*cur_w.dot(cur_w.vector, cur_w.vector, cur_w.vlen);
				ls.solution = cur_w;
				ls.gradient = cur_subgrad;
				SG_SPRINT("%lf\n", ls.fval)

				wbest_candidates.push_back(ls);
			}

			astar = ls_res[1].a * norm_dir;

			tstop=ttime.cur_time_diff(false);
			SG_SPRINT("\t\tline search time: %.5lf\n", tstop-tstart)
		}

		/* search for the best w among the new candidates */
		if (verbose)
			SG_SPRINT("\t searching for the best Fp:\n")
		for (size_t i = 0; i < wbest_candidates.size(); i++)
		{
			if (verbose)
				SG_SPRINT("\t\t %d fcurrent: %.16lf\n", i, wbest_candidates[i].fval)

			if (wbest_candidates[i].fval < best_Fp)
			{
				best_Fp = wbest_candidates[i].fval;
				best_risk = wbest_candidates[i].fval - wbest_candidates[i].reg;
				memcpy(best_w, wbest_candidates[i].solution.vector, sizeof(float64_t)*w_dim);
				memcpy(best_subgrad.vector, wbest_candidates[i].gradient.vector, sizeof(float64_t)*w_dim);

				ncbm.Fp = best_Fp;

				if (verbose)
					SG_SPRINT("\t\t new best norm: %f\n",
							best_w.twonorm(best_w.vector, w_dim));
			}

			if (!is_convex)
			{
				index_t cp_idx = ncbm.nCP-(wbest_candidates.size()-i);

				/* conflict */
				float64_t score
					= SGVector<float64_t>::dot(best_w.vector,
							wbest_candidates[i].gradient.vector, w_dim)
					+ (-1.0*bias[cp_idx]);
				if (score > best_risk)
				{
					float64_t U
						= best_risk
						- SGVector<float64_t>::dot(best_w.vector,
								wbest_candidates[i].gradient.vector, w_dim);

					float64_t L
						= best_Fp - wbest_candidates[i].reg
						- SGVector<float64_t>::dot(wbest_candidates[i].solution.vector,
								wbest_candidates[i].gradient.vector, w_dim);

					if (verbose)
						SG_SPRINT("CONFLICT Rbest=%.6lg score=%g L=%.6lg U=%.6lg\n", best_risk, score, L, U)
					if (L <= U)
					{
						if (verbose)
							SG_SPRINT("%.6lf < %.6lf => changing bias[%d]=%g\n", L, U, cp_idx, L)
						bias[cp_idx]= -L;
					}
					else
					{
						wbest_candidates[i].gradient.zero();
						SGVector<float64_t>::vec1_plus_scalar_times_vec2(wbest_candidates[i].gradient.vector, -_lambda, best_w.vector, w_dim);

						cp_ptr = CPList_tail;
						for (size_t j = wbest_candidates.size()-1; i < j; --j)
						{
							cp_ptr = cp_ptr->prev;
							SG_SPRINT("tail - %d\n (%d)", j, i)
						}

						float64_t* cp = get_cutting_plane(cp_ptr);
						LIBBMRM_MEMCPY(cp, wbest_candidates[i].gradient.vector, w_dim*sizeof(float64_t));

						/* update the corresponding column and row in H */
						cp_ptr = CPList_head;
						for (uint32_t j = 0; j < ncbm.nCP-1; ++j)
						{
							float64_t* a = get_cutting_plane(cp_ptr);
							cp_ptr = cp_ptr->next;
							float64_t dot_val
								= SGVector<float64_t>::dot(a, wbest_candidates[i].gradient.vector, w_dim);

							H.matrix[LIBBMRM_INDEX(cp_idx, j, maxCPs)]
								= H.matrix[LIBBMRM_INDEX(j, cp_idx, maxCPs)]
								= dot_val/_lambda;
						}

						diag_H[LIBBMRM_INDEX(cp_idx, cp_idx, maxCPs)]
							= SGVector<float64_t>::dot(wbest_candidates[i].gradient.vector,
									wbest_candidates[i].gradient.vector, w_dim);


						bias[cp_idx]
							= best_Fp - wbest_candidates[i].reg
							- SGVector<float64_t>::dot(wbest_candidates[i].solution.vector,
									wbest_candidates[i].gradient.vector, w_dim);

						if (verbose)
							SG_SPRINT("solved by changing nCP=%d bias:%g (%g)\n", cp_idx, bias[cp_idx], L)
					}
				}
			}
		}

		/* Inactive Cutting Planes (ICP) removal 
		if (cleanICP)
		{
			clean_icp(&icp_stats, ncbm, &CPList_head, &CPList_tail,
					H.matrix, diag_H.vector, x.vector,
					map.vector, cleanAfter, bias.vector, I.vector);
		}
		*/
	}

	memcpy(w, best_w.vector, sizeof(float64_t)*w_dim);

	/* free ICP_stats variables */
	LIBBMRM_FREE(icp_stats.ICPcounter);
	LIBBMRM_FREE(icp_stats.ICPs);
	LIBBMRM_FREE(icp_stats.ACPs);
	LIBBMRM_FREE(icp_stats.H_buff);

	return ncbm;
}

} /* namespace shogun */
