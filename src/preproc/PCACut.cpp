#include "PruneVarSubMean.h"
#include "RealPreProc.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

CPCACut::CPCACut()
  : CRealPreProc("PruneVarSubMean"), idx(NULL), num_idx(0)
{
}

CPCACut::~CPCACut()
{
  delete[] idx ;
}

/// initialize preprocessor from features
bool CPCACut::init(CFeatures* f_)
{
  CIO::message("calling CPCACut::init\n") ;

  CRealFeatures *f=(CRealFeatures*) f_ ;
  int num_examples=f->get_number_of_examples() ;
  int num_features=((CRealFeatures*)f)->get_num_features() ;
  double *mean=new double[num_features] ;
  double *var=new double[num_features] ;
  double *feature=NULL ;
  int j ;
  for (j=0; j<num_features; j++)
    {
      mean[j]=0 ; var[j]=0 ;
    } ;
  // compute mean
  for (int i=0; i<num_examples; i++)
    {
      long len ; bool free ;
      feature=f->get_feature_vector(i, len, free) ;

      for (int j=0; j<num_features; j++)
	mean[j]+=feature[j] ;

      f->free_feature_vector(feature, free) ;
      feature=NULL ;
    } ;

  for (j=0; j<num_features; j++)
    mean[j]/=num_examples ;

  // compute var
  for (int i=0; i<num_examples; i++)
    {
      long len ; bool free ;
      feature=f->get_feature_vector(i, len, free) ;

      for (int j=0; j<num_features; j++)
	var[j]+=(mean[j]-feature[j])*(mean[j]-feature[j]) ;

      f->free_feature_vector(feature, free) ;
      feature=NULL ;
    } ;

  int num_ok=0 ; int* idx_ok=new int[num_features] ;
  for (j=0; j<num_features; j++)
    {
      var[j]/=num_examples ;
      if (var[j]>1e-4) 
	{
	  idx_ok[num_ok]=j ;
	  num_ok++ ;
	} ;
    } ;
  //CIO::message("number of features: %i  number ok: %i\n", num_features, num_ok) ;
  delete[] idx ;
  idx=new int[num_ok] ;
  for (j=0; j<num_ok; j++)
    idx[j]=idx_ok[j] ;
  num_idx=num_ok ;
  delete[] idx_ok ;

  return true ;
}

/// initialize preprocessor from features
void CPCACut::cleanup()
{
  delete[] idx ;
  idx=NULL ;
}

/// initialize preprocessor from file
bool CPCACut::load(FILE* f)
{
  return false;
}

/// save preprocessor init-data to file
bool CPCACut::save(FILE* f)
{
  return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
REAL* CPCACut::apply_to_feature_matrix(CFeatures* f)
{
  return NULL;
}

/// apply preproc on single feature vector
/// result in feature matrix
REAL* CPCACut::apply_to_feature_vector(REAL* f, int &len)
{
  //CIO::message("preprocessing vector of length %i to length %i\n", len, num_idx) ;

  REAL *ret=new REAL[num_idx] ;
  for (int i=0; i<num_idx; i++)
    ret[i]=f[idx[i]] ;

  len=num_idx ;
  return ret;
}

