#include "features/ShortFeatures.h"
#include "preproc/ShortPreProc.h"

CShortFeatures::CShortFeatures()
: CFeatures(),  num_features(0), num_vectors(0),
  feature_matrix(NULL)
{
}

CShortFeatures::~CShortFeatures()
{
  delete[] feature_matrix;
}

/// get feature vector for sample num
short int* CShortFeatures::get_feature_vector(long num, long &len, bool &free)
{
  len=num_features ; 
  if ((num<num_vectors) && feature_matrix)
    {
      free=false ;
      return &feature_matrix[num*num_features];
    } 
  else
    {
      free=true ;
      short int* feat=new short int[len] ;
      compute_feature_vector(num, feat) ;
      if (preproc)
	{
	  short int* feat2 = ((CShortPreProc*) preproc)->apply_to_feature_vector(feat, len);
	  delete[] feat ;
	  return feat2 ;
	} ;
      return feat ;
    }
}

void CShortFeatures::free_feature_vector(short int* feat, bool free)
{
  if (free)
    delete[] feat ;
} 

short int* CShortFeatures::get_feature_matrix(long &num_feat, long &num_vec)
{
  num_feat=num_features;
  num_vec=num_vectors;
  return feature_matrix;
}

bool CShortFeatures::preproc_feature_matrix()
{
  if (preproc)
    ((CShortPreProc*) preproc)->apply_to_feature_matrix(this);
}
