#include "lib/common.h"
#include "lib/io.h"
#include "features/WordFeatures.h"
#include "features/Labels.h"
#include "distributions/hmm/LinearHMM.h"
#include "classifier/PluginEstimate.h"

#include <assert.h>

CPluginEstimate::CPluginEstimate() : pos_model(NULL), neg_model(NULL), test_features(NULL)
{
}

CPluginEstimate::~CPluginEstimate()
{
	delete pos_model;
	delete neg_model;
}

bool CPluginEstimate::train(CWordFeatures* features, CLabels* labels, REAL pos_pseudo_count, REAL neg_pseudo_count)
{
	delete pos_model;
	delete neg_model;

	pos_model=new CLinearHMM(features);
	neg_model=new CLinearHMM(features);

	INT* pos_indizes=new INT[((CWordFeatures*) features)->get_num_vectors()];
	INT* neg_indizes=new INT[((CWordFeatures*) features)->get_num_vectors()];

	assert(labels->get_num_labels() == features->get_num_vectors());

	INT pos_idx=0;
	INT neg_idx=0;

	for (INT i=0; i<labels->get_num_labels(); i++)
	{
		if (labels->get_label(i) > 0)
			pos_indizes[pos_idx++]=i;
		else
			neg_indizes[neg_idx++]=i;
	}
	//CIO::message("pos: %ld neg: %ld\n", pos_idx, neg_idx);

	CIO::message("training using pseudos %f and %f\n", pos_pseudo_count, neg_pseudo_count);
	pos_model->train(pos_indizes, pos_idx, pos_pseudo_count);
	neg_model->train(neg_indizes, neg_idx, neg_pseudo_count);

	delete[] pos_indizes;
	delete[] neg_indizes;
	
	return true;
}

REAL* CPluginEstimate::test()
{
	CWordFeatures* features=test_features;
	assert(features);

	if ((!pos_model) || (!neg_model))
	  {
	    CIO::message("model(s) not assigned\n") ;
	    return NULL ;
	  } ;

	REAL* result=new REAL[features->get_num_vectors()];
	assert(result);

	for (INT vec=0; vec<features->get_num_vectors(); vec++)
		result[vec]=classify_example(vec);

	return result;
}

CLabels* CPluginEstimate::classify(CLabels* result)
{
	CWordFeatures* features=test_features;

	assert(features);

	if (!result)
		result=new CLabels(features->get_num_vectors());

	assert(result);

	for (INT vec=0; vec<features->get_num_vectors(); vec++)
		result->set_label(vec, classify_example(vec));

	return result;
}

REAL CPluginEstimate::classify_example(INT idx)
{
	INT len;
	bool to_free;

	WORD* vector=((CWordFeatures*) test_features)->get_feature_vector(idx, len, to_free);

	if ((!pos_model) || (!neg_model))
	  {
	    CIO::message("model(s) not assigned\n") ;
	    return NAN ;
	  } ;
	  
	REAL result=pos_model->get_log_likelihood_example(vector, len) - neg_model->get_log_likelihood_example(vector, len);
	((CWordFeatures*) test_features)->free_feature_vector(vector, idx, to_free);
	return result;
}
