#include "Features.h"

CRealFeatures::CRealFeatures()
  : CFeatures(), num_vectors(0), num_features(0), feature_matrix(NULL)
{
}

CRealFeatures::~CRealFeatures()
{
  delete[] feature_matrix;
}

/// get feature vector for sample num
REAL* CRealFeatures::get_feature_vector(int num, int &len, bool &free)
{
  len=num_features ; 
  if ((num<num_vectors) && feature_matrix)
    {
      free=false ;
      return feature_matrix[num*num_features];
    } 
  else
    {
      free=true ;
      REAL* feat=get_feature_vector_comp(num) ;
      if (preproc)
	{
	  REAL* feat2 preproc->apply_to_feature_vector(feat, len);
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
const REAL* CRealFeatures::get_feature_matrix(int &num_feat, int &num_vect)
{
  num_feat=num_features;
  num_vec=num_vectors;
  return feature_matrix;
}

/// preproc feature_matrix
bool CRealFeatures::preproc_feature_matrix()
{
  if (preproc)
    preproc->preproc_feature(this);
}
