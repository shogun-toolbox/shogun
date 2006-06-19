/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"

#ifdef HAVE_LAPACK
#include "classifier/LDA.h"
#include "features/Labels.h"
#include "lib/Mathmatics.h"

CLDA::CLDA() : CLinearClassifier(), learn_rate(0.1), max_iter(10000000)
{
}


CLDA::~CLDA()
{
}

bool CLDA::train()
{
	DREAL prior=1.0;
	DREAL gamma=0;

	ASSERT(get_labels());
	ASSERT(get_features());
	INT num_train_labels=0;
	INT* train_labels=get_labels()->get_int_labels(num_train_labels);
	ASSERT(train_labels);

	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();
	ASSERT(num_vec==num_train_labels);

	INT* classidx_neg=new INT[num_vec];
	ASSERT(classidx_neg);
	INT* classidx_pos=new INT[num_vec];
	ASSERT(classidx_pos);

	INT i=0;
	INT j=0;
	INT num_neg=0;
	INT num_pos=0;
	for (i=0; i<num_train_labels; i++)
	{
		if (train_labels[i]==-1)
			classidx_neg[num_neg++]=i;
		else if (train_labels[i]==+1)
			classidx_pos[num_pos++]=i;
		else
		{
			CIO::message(M_ERROR, "found label != +/- 1 bailing...");
			return false;
		}
	}

	CIO::message(M_DEBUG,"num_neg: %d num_pos: %d\n", num_neg, num_pos);

	if (num_neg<=0 && num_pos<=0)
	{
		CIO::message(M_ERROR, "whooooo ? only a single class found\n");
		return false;
	}

	delete[] w;
	w=new DREAL[num_feat];
	ASSERT(w);

	DREAL* mean_neg=new DREAL[num_feat];
	ASSERT(mean_neg);
	memset(mean_neg,0,num_feat*sizeof(DREAL));

	DREAL* mean_pos=new DREAL[num_feat];
	ASSERT(mean_pos);
	memset(mean_pos,0,num_feat*sizeof(DREAL));

	DREAL* scatter=new DREAL[num_feat*num_feat];
	ASSERT(scatter);
	memset(scatter,0,num_feat*num_feat*sizeof(DREAL));

	DREAL* buffer=new DREAL[num_feat*CMath::max(num_neg, num_pos)];
	ASSERT(buffer);

	//neg
	for (i=0; i<num_neg; i++)
	{
		INT vlen;
		bool vfree;
		double* vec=features->get_feature_vector(classidx_neg[i], vlen, vfree);
		ASSERT(vec);

		for (j=0; j<vlen; j++)
		{
			mean_neg[j]+=vec[j];
			buffer[num_feat*i+j]=vec[j];
		}

		features->free_feature_vector(vec, classidx_neg[i], vfree);
	}

	for (j=0; j<num_feat; j++)
		mean_neg[j]/=num_neg;

	for (i=0; i<num_neg; i++)
	{
		for (j=0; j<num_feat; j++)
			buffer[num_feat*i+j]-=mean_neg[j];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_feat, num_feat, num_feat, 1.0, buffer, num_feat, buffer, num_feat, 1.0, scatter, num_feat);
	
	//mean pos
	for (i=0; i<num_pos; i++)
	{
		INT vlen;
		bool vfree;
		double* vec=features->get_feature_vector(classidx_pos[i], vlen, vfree);
		ASSERT(vec);

		for (j=0; j<vlen; j++)
		{
			mean_pos[j]+=vec[j];
			buffer[num_feat*i+j]=vec[j];
		}

		features->free_feature_vector(vec, classidx_pos[i], vfree);
	}

	for (j=0; j<num_feat; j++)
		mean_neg[j]/=num_neg;

	for (i=num_neg; i<num_train_labels; i++)
	{
		for (j=0; j<num_feat; j++)
			buffer[num_feat*i+j]-=mean_neg[j];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_feat, num_feat, num_feat, 1.0, buffer, num_feat, buffer, num_feat, 1.0, scatter, num_feat);

	DREAL trace=CMath::trace(scatter, num_feat, num_feat)/num_train_labels;

	for (i=0; i<num_feat*num_feat; i++)
		scatter[i]=(1-gamma)*scatter[i]/num_train_labels;

	for (i=0; i<num_feat; i++)
		scatter[i*num_feat+i]+= trace*gamma/num_feat;
	
	//DREAL* p= CMath::pinv(scatter, num_feat, num_feat, NULL);
	//bias=log(prior);
	//memcpy(buffer,mean_neg,sizeof(DREAL)*num_feat);
	//cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit, num_feat, scatter, num_feat, mean_neg, 1);
	//bias-=0.5*CMath::dot(mean_neg, buffer, num_feat);
	//cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit, num_feat, scatter, num_feat, mean_pos, 1);
	//bias+=0.5*CMath::dot(mean_pos, buffer, num_feat);

	for (i=0; i<num_feat; i++)
		w[i]=mean_pos[i]-mean_neg[i];

	delete[] train_labels;
	delete[] mean_neg;
	delete[] mean_pos;
	delete[] scatter;
	delete[] classidx_neg;
	delete[] classidx_pos;
	delete[] buffer;
	return false;
}

//priorP = ones(nClasses,1)/nClasses;
//
//d= size(xTr,1);
//m= zeros(d, nClasses);
//Sq= zeros(d, d);
//for ci= 1:nClasses,
//  cli= clInd{ci};
//  m(:,ci)= mean(xTr(:,cli),2);
//  yc= xTr(:,cli) - m(:,ci)*ones(1,N(ci));
//  Sq= Sq + yc*yc';
//end
//Sq= Sq/(sum(N)-1);
//Sq = (1-gamma)*Sq + gamma/d*trace(Sq)*eye(d);
//Sq = pinv(Sq);
//
//C.w = Sq*m;
//C.b = -0.5*sum(m.*C.w,1)' + log(priorP);
//
//if nClasses==2
//  C.w = C.w(:,2) - C.w(:,1);
//  C.b = C.b(2)-C.b(1);
//end
#endif
