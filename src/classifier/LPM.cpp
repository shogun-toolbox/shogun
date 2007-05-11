/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/LPM.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"

CLPM::CLPM() : CLinearClassifier(), learn_rate(0.1), max_iter(10000000)
{
}


CLPM::~CLPM()
{
}

bool CLPM::train()
{

//tr_ell = size(XT,2) ;
//Y_tr   = LT ;
//X_tr   = sparse([XT;-XT]) ;
//
//A = [sparse(-Y_tr'.*ones(tr_ell,1)) -speye(tr_ell) sparse(-X_tr*sparse(spdiag(Y_tr)))'] ;
//b = -ones(tr_ell,1) ;
//INF=1e20 ; 
//LB=[-INF;zeros(tr_ell,1); zeros(size(X_tr,1),1)] ;
//UB=[ INF;INF*ones(tr_ell,1); INF*ones(size(X_tr,1),1)];
//c=[0;PAR.C*ones(tr_ell,1)/tr_ell; ones(size(X_tr,1),1)] ;
//clear X_tr 
//
//disp('setting up problem');
//[p_lp,how]=lp_gen(lpenv,c,A,b,LB,UB,0,1) ;
//clear A c b LB UB Y_tr 
//tic
//  disp('solving problem');
//  [sol,lambda,how]=lp_resolve(lpenv,p_lp,1,'bar') ;
//toc
//if equal(how,'OK')
//  lp_close(lpenv,p_lp) ;
//else
//  how
//  keyboard
//end ;
//b     = sol(1) ;
//xis   = sol(2:tr_ell+1) ;
//alpha = sol(tr_ell+2:end);
//alpha = alpha(1:end/2)-alpha(end/2+1:end) ;
//sum(abs(alpha))

	ASSERT(get_labels());
	ASSERT(get_features());
	bool converged=false;
	INT iter=0;
	INT num_train_labels=0;
	INT* train_labels=get_labels()->get_int_labels(num_train_labels);
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	delete[] w;
	w=new DREAL[num_feat];
	w_dim=num_feat;
	ASSERT(w);
	DREAL* output=new DREAL[num_vec];
	ASSERT(output);

	//start with uniform w, bias=0
	bias=0;
	for (INT i=0; i<num_feat; i++)
		w[i]=1.0/num_feat;

	//loop till we either get everything classified right or reach max_iter
	while (!converged && iter<max_iter)
	{
		converged=true;
		for (INT i=0; i<num_vec; i++)
			output[i]=classify_example(i);

		for (INT i=0; i<num_vec; i++)
		{
			if (CMath::sign<DREAL>(output[i]) != train_labels[i])
			{
				converged=false;
				INT vlen;
				bool vfree;
				double* vec=features->get_feature_vector(i, vlen, vfree);

				bias+=learn_rate*train_labels[i];
				for (INT j=0; j<num_feat; j++)
					w[j]+=  learn_rate*train_labels[i]*vec[j];

				features->free_feature_vector(vec, i, vfree);
			}
		}

		iter++;
	}
	delete[] output;
	delete[] train_labels;

	return false;
}
