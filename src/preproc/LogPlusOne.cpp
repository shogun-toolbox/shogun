#include "LogPlusOne.h"
#include "RealPreProc.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/Mathmatics.h"

CLogPlusOne::CLogPlusOne() : CRealPreProc("LogPlusOne", "LG+1") 
{
}

CLogPlusOne::~CLogPlusOne()
{
}

/// initialize preprocessor from features
bool CLogPlusOne::init(CFeatures* f)
{
	assert(f->get_feature_class() == C_SIMPLE);
	assert(f->get_feature_type() == F_REAL);
	return true;
}

/// clean up allocated memory
void CLogPlusOne::cleanup()
{
}

/// initialize preprocessor from file
bool CLogPlusOne::load(FILE* f)
{
	return false;
}

/// save preprocessor init-data to file
bool CLogPlusOne::save(FILE* f)
{
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
REAL* CLogPlusOne::apply_to_feature_matrix(CFeatures* f)
{
	INT i,j;
	INT num_vec;
	INT num_feat;
	REAL* matrix=((CRealFeatures*) f)->get_feature_matrix(num_feat, num_vec);

	for (i=0; i<num_vec; i++)
	{
		REAL* vec=&matrix[i*num_feat];

		for (j=0; j<num_feat; j++)
			vec[j]=log(vec[j]+1);
	}
	return matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
REAL* CLogPlusOne::apply_to_feature_vector(REAL* f, INT& len)
{
	REAL* vec=new REAL[len];
	INT i=0;

	for (i=0; i<len; i++)
		vec[i]=log(f[i]+1);

	return vec;
}

/// initialize preprocessor from file
bool CLogPlusOne::load_init_data(FILE* src)
{
	return true;
}

/// save init-data (like transforamtion matrices etc) to file
bool CLogPlusOne::save_init_data(FILE* dst)
{
	return true;
}
