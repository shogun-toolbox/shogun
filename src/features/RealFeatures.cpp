#include <assert.h>
#include "features/RealFeatures.h"
#include "preproc/RealPreProc.h"
#include <string.h>

CRealFeatures::CRealFeatures() : CFeatures(), num_vectors(0), num_features(0), feature_matrix(NULL)
{
}

CRealFeatures::~CRealFeatures()
{
  delete[] feature_matrix;
}
  
CRealFeatures::CRealFeatures(const CRealFeatures & orig): CFeatures(orig), 
num_vectors(orig.num_vectors), num_features(orig.num_features)
{
	if (orig.feature_matrix)
	{
		feature_matrix=new REAL(num_vectors*num_features);
		memcpy(feature_matrix, orig.feature_matrix, num_vectors*num_features); 
	}
}

/// get feature vector for sample num
REAL* CRealFeatures::get_feature_vector(long num, long &len, bool &free)
{
	len=num_features; 
	assert(num<num_vectors);

	if (feature_matrix)
	{
		free=false ;
		return &feature_matrix[num*num_features];
	} 
	else
	{
		free=true ;
		REAL* feat=compute_feature_vector(num, len) ;
		if (preproc)
		{
			REAL* feat2 = ((CRealPreProc*) preproc)->apply_to_feature_vector(feat, len);
			delete[] feat ;
			return feat2 ;
		}
		return feat ;
	}
}

void CRealFeatures::free_feature_vector(REAL* feat, bool free)
{
  if (free)
    delete[] feat ;
} 

/// get the pointer to the feature matrix
REAL* CRealFeatures::get_feature_matrix(long &num_feat, long &num_vec)
{
  num_feat=num_features;
  num_vec=num_vectors;
  return feature_matrix;
}

/// preproc feature_matrix
bool CRealFeatures::preproc_feature_matrix()
{
	if (preproc)
#warning		return ((CRealPreProc*) preproc)->apply_to_feature_matrix(this);
	  return false ;
	else
		return false;
}
