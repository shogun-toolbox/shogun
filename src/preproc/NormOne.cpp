#include "NormOne.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

CNormOne::CNormOne()
{
}

CNormOne::~CNormOne()
{
}

/// initialize preprocessor from features
bool CNormOne::init(CFeatures* f)
{
	return true;
}

/// initialize preprocessor from features
void CNormOne::cleanup()
{
}

/// initialize preprocessor from file
bool CNormOne::load(FILE* f)
{
	return false;
}

/// save preprocessor init-data to file
bool CNormOne::save(FILE* f)
{
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
REAL* CNormOne::apply_to_feature_matrix(CFeatures* f)
{
	long i,j;
	long num_vec;
	long num_feat;
	REAL* matrix=((CRealFeatures*) f)->get_feature_matrix(num_feat, num_vec);

	for (i=0; i<num_vec; i++)
	{
		REAL sqnorm=0;
		REAL norm=0;
		REAL* vec=&matrix[i*num_feat];

		for (j=0; j<num_feat; j++)
			sqnorm+=vec[j]*vec[j];

		norm=sqrt(sqnorm);
		
		for (j=0; j<num_feat; j++)
			vec[j]/=norm;
	}
	return matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
REAL* CNormOne::apply_to_feature_vector(REAL* f, int len)
{
	REAL* vec=new REAL[len];
	REAL sqnorm=0;
	REAL norm=0;
	long i=0;

	for (i=0; i<len; i++)
		sqnorm+=f[i]*f[i];
	
	norm=sqrt(sqnorm);

	for (i=0; i<len; i++)
		vec[i]=f[i]/norm;

	return vec;
}

short int* CNormOne::apply_to_feature_vector(short int* f, int len)
{
	return NULL;
}


