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
#include "lib/io.h"
#include "features/StringFeatures.h"
#include "features/Labels.h"
#include "distributions/hmm/LinearHMM.h"
#include "classifier/PluginEstimate.h"


CPluginEstimate::CPluginEstimate() : CSGObject(), pos_model(NULL), neg_model(NULL), test_features(NULL)
{
}

CPluginEstimate::~CPluginEstimate()
{
	delete pos_model;
	delete neg_model;
}

bool CPluginEstimate::train(CStringFeatures<WORD>* features, CLabels* labels, DREAL pos_pseudo_count, DREAL neg_pseudo_count)
{
	delete pos_model;
	delete neg_model;

	pos_model=new CLinearHMM(features);
	neg_model=new CLinearHMM(features);

	INT* pos_indizes=new INT[((CStringFeatures<WORD>*) features)->get_num_vectors()];
	INT* neg_indizes=new INT[((CStringFeatures<WORD>*) features)->get_num_vectors()];

	ASSERT(labels->get_num_labels() == features->get_num_vectors());

	INT pos_idx=0;
	INT neg_idx=0;

	for (INT i=0; i<labels->get_num_labels(); i++)
	{
		if (labels->get_label(i) > 0)
			pos_indizes[pos_idx++]=i;
		else
			neg_indizes[neg_idx++]=i;
	}

	SG_INFO( "training using pseudos %f and %f\n", pos_pseudo_count, neg_pseudo_count);
	pos_model->train(pos_indizes, pos_idx, pos_pseudo_count);
	neg_model->train(neg_indizes, neg_idx, neg_pseudo_count);

	delete[] pos_indizes;
	delete[] neg_indizes;
	
	return true;
}

DREAL* CPluginEstimate::test()
{
	CStringFeatures<WORD>* features=test_features;
	ASSERT(features);

	if ((!pos_model) || (!neg_model))
	  {
       SG_ERROR( "model(s) not assigned\n");
	    return NULL ;
	  } ;

	DREAL* result=new DREAL[features->get_num_vectors()];
	ASSERT(result);

	for (INT vec=0; vec<features->get_num_vectors(); vec++)
		result[vec]=classify_example(vec);

	return result;
}

CLabels* CPluginEstimate::classify(CLabels* result)
{
	CStringFeatures<WORD>* features=test_features;

	ASSERT(features);

	if (!result)
		result=new CLabels(features->get_num_vectors());

	ASSERT(result);

	for (INT vec=0; vec<features->get_num_vectors(); vec++)
		result->set_label(vec, classify_example(vec));

	return result;
}

DREAL CPluginEstimate::classify_example(INT idx)
{
	INT len;

	WORD* vector=((CStringFeatures<WORD>*) test_features)->get_feature_vector(idx, len);

	if ((!pos_model) || (!neg_model))
	  {
       SG_ERROR( "model(s) not assigned\n");
	    return NAN;
	  } ;
	  
	DREAL result=pos_model->get_log_likelihood_example(vector, len) - neg_model->get_log_likelihood_example(vector, len);
	return result;
}
