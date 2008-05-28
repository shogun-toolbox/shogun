/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"

#ifdef HAVE_LAPACK
#include "classifier/Classifier.h"
#include "classifier/LinearClassifier.h"
#include "classifier/LDA.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "lib/lapack.h"

CLDA::CLDA(DREAL gamma)
: CLinearClassifier(), m_gamma(gamma)
{
}

CLDA::CLDA(DREAL gamma, CRealFeatures* traindat, CLabels* trainlab)
: CLinearClassifier(), m_gamma(gamma)
{
	set_features(traindat);
	set_labels(trainlab);
}


CLDA::~CLDA()
{
}

bool CLDA::train()
{
	ASSERT(labels);
	ASSERT(features);
	INT num_train_labels=0;
	INT* train_labels=labels->get_int_labels(num_train_labels);
	ASSERT(train_labels);

	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();
	ASSERT(num_vec==num_train_labels);

	INT* classidx_neg=new INT[num_vec];
	INT* classidx_pos=new INT[num_vec];

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
			SG_ERROR( "found label != +/- 1 bailing...");
			return false;
		}
	}

	if (num_neg<=0 && num_pos<=0)
	{
      SG_ERROR( "whooooo ? only a single class found\n");
		return false;
	}

	delete[] w;
	w=new DREAL[num_feat];
	w_dim=num_feat;

	DREAL* mean_neg=new DREAL[num_feat];
	memset(mean_neg,0,num_feat*sizeof(DREAL));

	DREAL* mean_pos=new DREAL[num_feat];
	memset(mean_pos,0,num_feat*sizeof(DREAL));

	DREAL* scatter=new DREAL[num_feat*num_feat];
	DREAL* buffer=new DREAL[num_feat*CMath::max(num_neg, num_pos)];

	//mean neg
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
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, num_feat, num_feat, num_neg, 1.0, buffer, num_feat, buffer, num_feat, 0, scatter, num_feat);
	
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
		mean_pos[j]/=num_pos;

	for (i=0; i<num_pos; i++)
	{
		for (j=0; j<num_feat; j++)
			buffer[num_feat*i+j]-=mean_pos[j];
	}
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, num_feat, num_feat, num_pos, 1.0/(num_train_labels-1), buffer, num_feat, buffer, num_feat, 1.0/(num_train_labels-1), scatter, num_feat);

	DREAL trace=CMath::trace(scatter, num_feat, num_feat);

	double s=1.0-m_gamma;

	for (i=0; i<num_feat*num_feat; i++)
		scatter[i]*=s;

	for (i=0; i<num_feat; i++)
		scatter[i*num_feat+i]+= trace*m_gamma/num_feat;

	DREAL* inv_scatter= CMath::pinv(scatter, num_feat, num_feat, NULL);

	DREAL* w_pos=buffer;
	DREAL* w_neg=&buffer[num_feat];

	cblas_dsymv(CblasColMajor, CblasUpper, num_feat, 1.0, inv_scatter, num_feat, mean_pos, 1, 0, w_pos, 1);
	cblas_dsymv(CblasColMajor, CblasUpper, num_feat, 1.0, inv_scatter, num_feat, mean_neg, 1, 0, w_neg, 1);
	
	bias=0.5*(CMath::dot(w_neg, mean_neg, num_feat)-CMath::dot(w_pos, mean_pos, num_feat));
	for (i=0; i<num_feat; i++)
		w[i]=w_pos[i]-w_neg[i];

#ifdef DEBUG_LDA
	SG_PRINT("bias: %f\n", bias);
    CMath::display_vector(w, num_feat, "w");
    CMath::display_vector(w_pos, num_feat, "w_pos");
    CMath::display_vector(w_neg, num_feat, "w_neg");
    CMath::display_vector(mean_pos, num_feat, "mean_pos");
    CMath::display_vector(mean_neg, num_feat, "mean_neg");
#endif

	delete[] train_labels;
	delete[] mean_neg;
	delete[] mean_pos;
	delete[] scatter;
	delete[] inv_scatter;
	delete[] classidx_neg;
	delete[] classidx_pos;
	delete[] buffer;
	return true;
}
#endif
