/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2008 Chun-Nam Yu
 */

#include <shogun/structure/CCSOSVM.h>
#include <shogun/mathematics/Mosek.h>
#include <shogun/lib/SGSparseVector.h>

using namespace shogun;

CCCSOSVM::CCCSOSVM()
	: CLinearStructuredOutputMachine()
{
	init();
}

CCCSOSVM::CCCSOSVM(CStructuredModel* model, SGVector<float64_t> w)
	: CLinearStructuredOutputMachine(model, model->get_labels())
{
	init();

	if (w.vlen)
	{
		set_w(w);
	}
	else
	{
		m_w.resize_vector(m_model->get_dim());
		m_w.zero();
	}
}

CCCSOSVM::~CCCSOSVM()
{
#ifdef USE_MOSEK
	MSK_deleteenv(&m_msk_env);
#endif
}

int32_t CCCSOSVM::mosek_qp_optimize(float64_t** G, float64_t* delta, float64_t* alpha, int32_t k, float64_t* dual_obj, float64_t rho)
{
#ifdef USE_MOSEK
	int32_t t;
	index_t Q_size = k*(k+1)/2;
	SGVector<float64_t> c(k);
	MSKlidxt *aptrb;
	MSKlidxt *aptre;
	MSKidxt *asub;
	SGVector<float64_t> aval(k);
	MSKboundkeye bkc[1];
	float64_t blc[1];
	float64_t buc[1];
	MSKboundkeye *bkx;
	SGVector<float64_t> blx(k);
	SGVector<float64_t> bux(k);
	MSKidxt *qsubi,*qsubj;
	SGVector<float64_t> qval(Q_size);

	MSKtask_t task;
	MSKrescodee r;

	aptrb = (MSKlidxt*) SG_MALLOC(MSKlidxt, k);
	aptre = (MSKlidxt*) SG_MALLOC(MSKlidxt, k);
	asub = (MSKidxt*) SG_MALLOC(MSKidxt, k);
	bkx = (MSKboundkeye*) SG_MALLOC(MSKboundkeye, k);
	qsubi = (MSKidxt*) SG_MALLOC(MSKidxt, Q_size);
	qsubj = (MSKidxt*) SG_MALLOC(MSKidxt, Q_size);


	/* DEBUG */
	/*
	 for (int32_t i=0;i<k;i++)
		 printf("delta: %.4f\n", delta[i]);

	 printf("G:\n");
	 for (int32_t i=0;i<k;i++)
	 {
		 for (int32_t j=0;j<k;j++)
		 	printf("%.4f ", G[i][j]);
		 printf("\n");
	 }
	 fflush(stdout);
	*/
	/* DEBUG */

	for (int32_t i=0; i < k;i++)
	{
		c[i] = -delta[i];
		aptrb[i] = i;
		aptre[i] = i+1;
		asub[i] = 0;
		aval[i] = 1.0;
		bkx[i] = MSK_BK_LO;
		blx[i] = 0.0;
		bux[i] = MSK_INFINITY;
	}
	bkc[0] = MSK_BK_FX;
	blc[0] = m_C;
	buc[0] = m_C;
	/*
	bkc[0] = MSK_BK_UP;
	blc[0] = -MSK_INFINITY;
	buc[0] = m_C;
	*/

	/* create the optimization task */
	r = MSK_maketask(m_msk_env, 1, k, &task);

	if (r != MSK_RES_OK)
		SG_ERROR("Could not create MOSEK task: %d\n", r)

	r = MSK_inputdata(task,
			1,k,
			1,k,
			c,0.0,
			aptrb,aptre,
			asub,aval,
			bkc,blc,buc,
			bkx,blx,bux);

	if (r != MSK_RES_OK)
		SG_ERROR("Error setting input data: %d\n", r)

	/* coefficients for the Gram matrix */
	t = 0;
	for (int32_t i=0;i<k;i++)
	{
		for (int32_t j=0;j<=i;j++)
		{
			qsubi[t] = i;
			qsubj[t] = j;
			qval[t] = G[i][j]/(1+rho);
			t++;
		}
	}

	r = MSK_putqobj(task, k*(k+1)/2, qsubi, qsubj, qval);
	if (r != MSK_RES_OK)
		SG_ERROR("Error MSK_putqobj: %d\n", r)

	/* DEBUG */
	/*
	 printf("t: %ld\n", t);
	 for (int32_t i=0;i<t;i++) {
	 printf("qsubi: %d, qsubj: %d, qval: %.4f\n", qsubi[i], qsubj[i], qval[i]);
	 }
	 fflush(stdout);
	 */
	/* DEBUG */

	/* set relative tolerance gap (DEFAULT = 1E-8)*/
	MSK_putdouparam(task, MSK_DPAR_INTPNT_TOL_REL_GAP, 1E-8);

	r = MSK_optimize(task);

	if (r != MSK_RES_OK)
		SG_ERROR("Error MSK_optimize: %d\n", r)

	MSK_getsolutionslice(task,
			MSK_SOL_ITR,
			MSK_SOL_ITEM_XX,
			0,
			k,
			alpha);

	/* output the objective value */
	MSK_getprimalobj(task, MSK_SOL_ITR, dual_obj);

	MSK_deletetask(&task);

	/* free the memory */
	SG_FREE(aptrb);
	SG_FREE(aptre);
	SG_FREE(asub);
	SG_FREE(bkx);
	SG_FREE(qsubi);
	SG_FREE(qsubj);

	return r;
#else
	return -1;
#endif
}

bool CCCSOSVM::train_machine(CFeatures* data)
{
	if (data)
		set_features(data);

	SGVector<float64_t> alpha;
	float64_t** G; /* Gram matrix */
	DynArray<SGSparseVector<float64_t> > dXc; /* constraint matrix */
	//	DOC **dXc; /* constraint matrix */
	SGVector<float64_t> delta; /* rhs of constraints */
	SGSparseVector<float64_t> new_constraint;
	float64_t dual_obj=0, alphasum;
	int32_t iter, size_active;
	float64_t value;
	SGVector<int32_t> idle; /* for cleaning up */
	float64_t margin;
	float64_t primal_obj;
	SGVector<float64_t> proximal_rhs;
	SGVector<float64_t> gammaG0;
	float64_t min_rho = 0.001;
	float64_t serious_counter=0;
	float64_t rho = 1.0; /* temporarily set it to 1 first */

	float64_t expected_descent, primal_obj_b=-1, reg_master_obj;
	int32_t null_step=1;
	float64_t kappa=0.1;
	float64_t temp_var;
	float64_t proximal_term, primal_lower_bound;

	float64_t v_k;
	float64_t obj_difference;
	SGVector<float64_t> cut_error; // cut_error[i] = alpha_{k,i} at current center x_k
	float64_t sigma_k;
	float64_t m2 = 0.2;
	float64_t m3 = 0.9;
	float64_t gTd;
	float64_t last_sigma_k=0;

	float64_t initial_primal_obj;
	int32_t suff_decrease_cond=0;
	float64_t decrease_proportion = 0.2; // start from 0.2 first

	float64_t z_k_norm;
	float64_t last_z_k_norm=0;

	/* warm start */
	SGVector<float64_t> w_b = m_w.clone();

	iter = 0;
	size_active = 0;
	G = NULL;

	new_constraint = find_cutting_plane(&margin);
	value = margin - new_constraint.dense_dot(1.0, m_w.vector, m_w.vlen, 0);

	primal_obj_b = primal_obj = 0.5*m_w.dot(m_w.vector, m_w.vector, m_w.vlen)+m_C*value;
	primal_lower_bound = 0;
	expected_descent = -primal_obj_b;
	initial_primal_obj = primal_obj_b;

	SG_INFO("Running CCCP inner loop solver: ")

	while ((!suff_decrease_cond) && (expected_descent<-m_eps) && (iter<m_max_iter))
	{
		++iter;
		++size_active;

		SG_DEBUG("ITER %d\n", iter)
		SG_PRINT(".")

		/* add constraint */
		dXc.resize_array(size_active);
		dXc[size_active - 1] = new_constraint;
		//		dXc[size_active - 1].add(new_constraint);
		/*
			 dXc = (DOC**)realloc(dXc, sizeof(DOC*)*size_active);
			 dXc[size_active-1] = (DOC*)malloc(sizeof(DOC));
			 dXc[size_active-1]->fvec = new_constraint;
			 dXc[size_active-1]->slackid = 1; // only one common slackid (one-slack)
			 dXc[size_active-1]->costfactor = 1.0;
			 */
		delta.resize_vector(size_active);
		delta[size_active-1] = margin;
		alpha.resize_vector(size_active);
		alpha[size_active-1] = 0.0;
		idle.resize_vector(size_active);
		idle[size_active-1] = 0;
		/* proximal point */
		proximal_rhs.resize_vector(size_active);
		cut_error.resize_vector(size_active);
		// note g_i = - new_constraint
		cut_error[size_active-1] = m_C*(new_constraint.dense_dot(1.0, w_b.vector, w_b.vlen, 0) - new_constraint.dense_dot(1.0, m_w.vector, m_w.vlen, 0));
		cut_error[size_active-1] += (primal_obj_b - 0.5*w_b.dot(w_b.vector, w_b.vector, w_b.vlen));
		cut_error[size_active-1] -= (primal_obj - 0.5*m_w.dot(m_w.vector, m_w.vector, m_w.vlen));

		gammaG0.resize_vector(size_active);

		/* update Gram matrix */
		G = SG_REALLOC(float64_t*, G, size_active-1, size_active);
		G[size_active - 1] = NULL;
		for (index_t j=0; j < size_active;j++)
		{
			G[j] = SG_REALLOC(float64_t, G[j], size_active-1, size_active);
		}
		for (index_t j=0; j < size_active-1; j++)
		{
			G[size_active-1][j] = dXc[size_active-1].sparse_dot(dXc[j]);
			G[j][size_active-1] = G[size_active-1][j];
		}
		G[size_active-1][size_active-1] = dXc[size_active-1].sparse_dot(dXc[size_active-1]);

		/* update gammaG0 */
		if (null_step==1)
		{
			gammaG0[size_active-1] = dXc[size_active-1].dense_dot(1.0, w_b.vector, w_b.vlen, 0);
		}
		else
		{
			for (index_t i = 0; i < size_active; i++)
				gammaG0[i] = dXc[i].dense_dot(1.0, w_b.vector, w_b.vlen, 0);
		}

		/* update proximal_rhs */
		for (index_t i = 0; i < size_active; i++)
		{
			switch(m_qp_type)
			{
				case MOSEK:
					proximal_rhs[i] = delta[i] - rho/(1+rho)*gammaG0[i];
					break;
				case SVMLIGHT:
					proximal_rhs[i] = (1+rho)*delta[i] - rho*gammaG0[i];
					break;
				default:
					SG_ERROR("Invalid QPType: %d\n", m_qp_type)
			}
		}

		switch(m_qp_type)
		{
			case MOSEK:
				/* solve QP to update alpha */
				dual_obj = 0;
				mosek_qp_optimize(G, proximal_rhs.vector, alpha.vector, size_active, &dual_obj, rho);
				break;
			case SVMLIGHT:
				/* TODO: port required functionality from the latest SVM^light into shogun
				 * in order to be able to support this
				 *
				if (size_active>1)
				{
					if (svmModel!=NULL)
						free_model(svmModel,0);
					svmModel = (MODEL*)my_malloc(sizeof(MODEL));
					svm_learn_optimization(dXc,proximal_rhs,size_active,sm->sizePsi,&lparm,&kparm,NULL,svmModel,alpha);
				}
				else
				{
					ASSERT(size_active==1)
					alpha[0] = m_C;
				}
				*/
				break;
			default:
				SG_ERROR("Invalid QPType: %d\n", m_qp_type)
		}

		/* DEBUG */
		//printf("r: %d\n", r); fflush(stdout);
		//printf("dual: %.16lf\n", dual_obj);
		/* END DEBUG */

		m_w.zero();
		for (index_t j = 0; j < size_active; j++)
		{
			if (alpha[j]>m_C*m_alpha_thrld)
			{
				// TODO: move this to SGVector
				// basically it's vector[i]= scale*sparsevector[i]
				for (index_t k = 0; k < dXc[j].num_feat_entries; k++)
				{
					index_t idx = dXc[j].features[k].feat_index;
					m_w[idx] += alpha[j]/(1+rho)*dXc[j].features[k].entry;
				}
			}
		}

		if (m_qp_type == SVMLIGHT)
		{
			/* compute dual obj */
			dual_obj = +0.5*(1+rho)*m_w.dot(m_w.vector, m_w.vector, m_w.vlen);
			for (int32_t j=0;j<size_active;j++)
				dual_obj -= proximal_rhs[j]/(1+rho)*alpha[j];
		}

		z_k_norm = CMath::sqrt(m_w.dot(m_w.vector, m_w.vector, m_w.vlen));
		m_w.vec1_plus_scalar_times_vec2(m_w.vector, rho/(1+rho), w_b.vector, w_b.vlen);

		/* detect if step size too small */
		sigma_k = 0;
		alphasum = 0;
		for (index_t j = 0; j < size_active; j++)
		{
			sigma_k += alpha[j]*cut_error[j];
			alphasum+=alpha[j];
		}
		sigma_k/=m_C;
		gTd = -m_C*(new_constraint.dense_dot(1.0, m_w.vector, m_w.vlen, 0)
				- new_constraint.dense_dot(1.0, w_b.vector, w_b.vlen, 0));

		for (index_t j = 0; j < size_active; j++)
			SG_DEBUG("alpha[%d]: %.8g, cut_error[%d]: %.8g\n", j, alpha[j], j, cut_error[j])
		SG_DEBUG("sigma_k: %.8g\n", sigma_k)
		SG_DEBUG("alphasum: %.8g\n", alphasum)
		SG_DEBUG("g^T d: %.8g\n", gTd)

		/* update cleanup information */
		for (index_t j = 0; j < size_active; j++)
		{
			if (alpha[j]<m_alpha_thrld*m_C)
				idle[j]++;
			else
				idle[j]=0;
		}

		new_constraint = find_cutting_plane(&margin);
		value = margin - new_constraint.dense_dot(1.0, m_w.vector, m_w.vlen, 0);

		/* print primal objective */
		primal_obj = 0.5*m_w.dot(m_w.vector, m_w.vector, m_w.vlen)+m_C*value;

		SG_DEBUG("ITER PRIMAL_OBJ %.4f\n", primal_obj)

		temp_var = w_b.dot(w_b.vector, w_b.vector, w_b.vlen);
		proximal_term = 0.0;
		for (index_t i=0; i < m_model->get_dim(); i++)
			proximal_term += (m_w[i]-w_b[i])*(m_w[i]-w_b[i]);

		reg_master_obj = -dual_obj+0.5*rho*temp_var/(1+rho);
		expected_descent = reg_master_obj - primal_obj_b;

		v_k = (reg_master_obj - proximal_term*rho/2) - primal_obj_b;

		primal_lower_bound = CMath::max(primal_lower_bound, reg_master_obj - 0.5*rho*(1+rho)*proximal_term);

		SG_DEBUG("ITER REG_MASTER_OBJ: %.4f\n", reg_master_obj)
		SG_DEBUG("ITER EXPECTED_DESCENT: %.4f\n", expected_descent)
		SG_DEBUG("ITER PRIMLA_OBJ_B: %.4f\n", primal_obj_b)
		SG_DEBUG("ITER RHO: %.4f\n", rho)
		SG_DEBUG("ITER ||w-w_b||^2: %.4f\n", proximal_term)
		SG_DEBUG("ITER PRIMAL_LOWER_BOUND: %.4f\n", primal_lower_bound)
		SG_DEBUG("ITER V_K: %.4f\n", v_k)
		SG_DEBUG("ITER margin: %.4f\n", margin)
		SG_DEBUG("ITER psi*-psi: %.4f\n", value-margin)

		obj_difference = primal_obj - primal_obj_b;

		if (primal_obj<primal_obj_b+kappa*expected_descent)
		{
			/* extra condition to be met */
			if ((gTd>m2*v_k)||(rho<min_rho+1E-8))
			{
				SG_DEBUG("SERIOUS STEP\n")

				/* update cut_error */
				for (index_t i = 0; i < size_active; i++)
				{
					cut_error[i] -= (primal_obj_b - 0.5*w_b.dot(w_b.vector, w_b.vector, w_b.vlen));
					cut_error[i] -= m_C*dXc[i].dense_dot(1.0, w_b.vector, w_b.vlen, 0);
					cut_error[i] += (primal_obj - 0.5*m_w.dot(m_w, m_w, m_w.vlen));
					cut_error[i] += m_C*dXc[i].dense_dot(1.0, m_w.vector, m_w.vlen, 0);
				}
				primal_obj_b = primal_obj;
				/* copy w_b <- m_w */
				for (index_t i=0; i < m_model->get_dim(); i++)
				{
					w_b[i] = m_w[i];
				}
				null_step = 0;
				serious_counter++;
			}
			else
			{
				/* increase step size */
				SG_DEBUG("NULL STEP: SS(ii) FAILS.\n")

				serious_counter--;
				rho = CMath::max(rho/10,min_rho);
			}
		}
		else
		{ /* no sufficient decrease */
			serious_counter--;

			if ((cut_error[size_active-1]>m3*last_sigma_k)&&(CMath::abs(obj_difference)>last_z_k_norm+last_sigma_k))
			{
				SG_DEBUG("NULL STEP: NS(ii) FAILS.\n")
				rho = CMath::min(10*rho,m_max_rho);
			}
			else
				SG_DEBUG("NULL STEP\n")
		}
		/* update last_sigma_k */
		last_sigma_k = sigma_k;
		last_z_k_norm = z_k_norm;


		/* break away from while loop if more than certain proportioal decrease in primal objective */
		if (primal_obj_b/initial_primal_obj<1-decrease_proportion)
			suff_decrease_cond = 1;

		/* clean up */
		if (iter % m_cleanup_check == 0)
		{
			size_active = resize_cleanup(size_active, idle, alpha, delta, gammaG0, proximal_rhs, &G, dXc, cut_error);
			ASSERT(size_active == proximal_rhs.vlen)
		}
	} // end cutting plane while loop

	SG_INFO(" Inner loop optimization finished.\n")

	for (index_t j = 0; j < size_active; j++)
		SG_FREE(G[j]);
	SG_FREE(G);

	/* copy */
	for (index_t i=0; i < m_model->get_dim(); i++)
		m_w[i] = w_b[i];

	m_primal_obj = primal_obj_b;

	return true;
}

SGSparseVector<float64_t> CCCSOSVM::find_cutting_plane(float64_t* margin)
{
	SGVector<float64_t> new_constraint(m_model->get_dim());
	int32_t psi_size = m_model->get_dim();

	index_t num_samples = m_model->get_features()->get_num_vectors();
	/* find cutting plane */
	*margin = 0;
	new_constraint.zero();
	for (index_t i = 0; i < num_samples; i++)
	{
		CResultSet* result = m_model->argmax(m_w, i);
		if (!result->psi_computed_sparse)
		{
			new_constraint.add(result->psi_truth);
			result->psi_pred.scale(-1.0);
			new_constraint.add(result->psi_pred);
		}
		else
		{
			new_constraint.add(result->psi_truth_sparse.get_dense(psi_size));
			SGVector<float64_t> psi_pred_dense =
				result->psi_pred_sparse.get_dense(psi_size);
			psi_pred_dense.scale(-1.0);
			new_constraint.add(psi_pred_dense);
		}
		/*
		printf("%.16lf %.16lf\n",
				SGVector<float64_t>::dot(result->psi_truth.vector, result->psi_truth.vector, result->psi_truth.vlen),
				SGVector<float64_t>::dot(result->psi_pred.vector, result->psi_pred.vector, result->psi_pred.vlen));
		*/
		*margin += result->delta;
		SG_UNREF(result);
	}
	/* scaling */
	float64_t scale = 1/(float64_t)num_samples;
	new_constraint.scale(scale);
	*margin *= scale;

	/* find the nnz elements in new_constraint */
	index_t l = 0;
	for (index_t i=0; i < psi_size; i++)
	{
		if (CMath::abs(new_constraint[i])>1E-10)
			l++; // non-zero
	}
	/* TODO: does this really work good?
		 l = CMath::get_num_nonzero(new_constraint.vector, new_constraint.vlen);
		 */
	/* create a sparse vector of the nnz of new_constraint */
	SGSparseVector<float64_t> cut_plane(l);
	index_t k = 0;
	for (index_t i = 0; i < psi_size; i++)
	{
		if (CMath::abs(new_constraint[i])>1E-10)
		{
			cut_plane.features[k].feat_index = i;
			cut_plane.features[k].entry = new_constraint[i];
			k++;
		}
	}

	return cut_plane;
}

int32_t CCCSOSVM::resize_cleanup(int32_t size_active, SGVector<int32_t>& idle, SGVector<float64_t>&alpha,
		SGVector<float64_t>& delta, SGVector<float64_t>& gammaG0,
		SGVector<float64_t>& proximal_rhs, float64_t ***ptr_G,
		DynArray<SGSparseVector<float64_t> >& dXc, SGVector<float64_t>& cut_error)
{
	int32_t i, j, new_size_active;
	index_t k;

	float64_t **G = *ptr_G;

	i=0;
	while ((i<size_active)&&(idle[i]<m_idle_iter))
		i++;

	j=i;
	while((j<size_active)&&(idle[j]>=m_idle_iter))
		j++;

	while (j<size_active)
	{
		/* copying */
		alpha[i] = alpha[j];
		delta[i] = delta[j];
		gammaG0[i] = gammaG0[j];
		cut_error[i] = cut_error[j];

		SG_FREE(G[i]);
		G[i] = G[j];
		G[j] = NULL;
	//	free_example(dXc[i],0);
		dXc[i] = dXc[j];
//		dXc[j] = NULL;

		i++;
		j++;
		while((j<size_active)&&(idle[j]>=m_idle_iter))
			j++;
	}
	for (k=i;k<size_active;k++)
	{
		if (G[k]!=NULL) SG_FREE(G[k]);
//		if (dXc[k].num_feat_entries > 0) SG_UNREF(dXc[k]);
	}
	new_size_active = i;
	alpha.resize_vector(new_size_active);
	delta.resize_vector(new_size_active);
	gammaG0.resize_vector(new_size_active);
	proximal_rhs.resize_vector(new_size_active);
	G = SG_REALLOC(float64_t*, G, size_active, new_size_active);
	dXc.resize_array(new_size_active);
	cut_error.resize_vector(new_size_active);

	/* resize G and idle */
	i=0;
	while ((i<size_active)&&(idle[i]<m_idle_iter))
		i++;

	j=i;
	while((j<size_active)&&(idle[j]>=m_idle_iter))
		j++;

	while (j<size_active)
	{
		idle[i] = idle[j];
		for (k=0;k<new_size_active;k++)
			G[k][i] = G[k][j];

		i++;
		j++;
		while((j<size_active)&&(idle[j]>=m_idle_iter))
			j++;
	}
	idle.resize_vector(new_size_active);
	for (k=0;k<new_size_active;k++)
		G[k] = SG_REALLOC(float64_t, G[k], size_active, new_size_active);

	*ptr_G = G;

	return new_size_active;
}

void CCCSOSVM::init()
{
	m_C = 1.0;
	m_eps = 1E-3;
	m_alpha_thrld = 1E-14;
	m_cleanup_check = 100;
	m_idle_iter = 20;
	m_max_iter = 1000;
	m_max_rho = m_C;
	m_primal_obj = CMath::INFTY;
	m_qp_type = MOSEK;

#ifdef USE_MOSEK
	MSKrescodee r = MSK_RES_OK;

	/* create mosek environment */
#if (MSK_VERSION_MAJOR == 6)
	r = MSK_makeenv(&m_msk_env, NULL, NULL, NULL, NULL);
#elif (MSK_VERSION_MAJOR == 7)
	r = MSK_makeenv(&m_msk_env, NULL);
#else
	#error "Unsupported Mosek version"
#endif

	/* check return code */
	if (r != MSK_RES_OK)
		SG_ERROR("Error while creating mosek env: %d\n", r)

	/* initialize the environment */
	r = MSK_initenv(m_msk_env);
	if (r != MSK_RES_OK)
		SG_ERROR("Error while initializing mosek env: %d\n", r)
#endif

	SG_ADD(&m_C, "m_C", "C", MS_NOT_AVAILABLE);
	SG_ADD(&m_eps, "m_eps", "Epsilon", MS_NOT_AVAILABLE);
	SG_ADD(&m_alpha_thrld, "m_alpha_thrld", "Alpha threshold", MS_NOT_AVAILABLE);
	SG_ADD(&m_cleanup_check, "m_cleanup_check", "Cleanup after given number of iterations", MS_NOT_AVAILABLE);
	SG_ADD(&m_idle_iter, "m_idle_iter", "Maximum number of idle iteration", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iter, "m_max_iter", "Maximum number of iterations", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_rho, "m_max_rho", "Max rho", MS_NOT_AVAILABLE);
	SG_ADD(&m_primal_obj, "m_primal_obj", "Primal objective value", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &m_qp_type, "m_qp_type", "QP Solver Type", MS_NOT_AVAILABLE);
}

EMachineType CCCSOSVM::get_classifier_type()
{
	return CT_CCSOSVM;
}
