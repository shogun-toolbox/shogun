#include "PruneVarSubMean.h"
#include "RealPreProc.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

CPruneVarSubMean::CPruneVarSubMean()
{
}

CPruneVarSubMean::~CPruneVarSubMean()
{
}

/// initialize preprocessor from features
bool CPruneVarSubMean::init(CFeatures* f)
{
	return true;
}

/// initialize preprocessor from features
void CPruneVarSubMean::cleanup()
{
}

/// initialize preprocessor from file
bool CPruneVarSubMean::load(FILE* f)
{
	return false;
}

/// save preprocessor init-data to file
bool CPruneVarSubMean::save(FILE* f)
{
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
REAL* CPruneVarSubMean::apply_to_feature_matrix(CFeatures* f)
{
	return NULL;
}

/// apply preproc on single feature vector
/// result in feature matrix
REAL* CPruneVarSubMean::apply_to_feature_vector(REAL* f, int len)
{
	return NULL;
}

#warning TODO implement
#ifdef PRUNE_VAR_SUB_MEAN

void CHMM::subtract_mean_from_top_feature_cache(int num_features, int totobs)
{
	if (feature_cache_obs)
	{
		for (int j=0; j<num_features; j++)
		{
			double mean=0;
			for (int i=0; i<totobs; i++)
				mean+=feature_cache_obs[i*num_features+j];
			for (int i=0; i<totobs; i++)
				feature_cache_obs[i*num_features+j]-=mean;
		}
	}
}

#endif
