#include "PCACut.h"
#include "RealPreProc.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"


extern "C" void cleaner_main(double *covZ, int dim, double thresh,
			     double **T, int *num_dim)  ;

CPCACut::CPCACut()
  : CRealPreProc("PruneVarSubMean"), T(NULL), num_dim(0)
{
}

CPCACut::~CPCACut()
{
  delete[] T ;
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

  CIO::message("computing covariance matrix...") ;

  double *cov=new double[num_features*num_features] ;

  for (int i=0; i<num_examples; i++)
    {
      long len ; bool free ;
      feature=f->get_feature_vector(i, len, free) ;

      for (int j=0; j<num_features; j++)
	feature[j]-=mean[j] ;

      // dger_(num_features,num_features, 1.0, feature, 1, feature, 1, cov, num_features) ;
      for (int k=0; k<num_features; k++)
	for (int l=0; l<num_features; l++)
	  cov[k*num_features+l]+=feature[l]*feature[k] ;

      f->free_feature_vector(feature, free) ;
      feature=NULL ;
    } ;

  for (int k=0; k<num_features; k++)
    for (int l=0; l<num_features; l++)
      cov[k*num_features+l]/=num_examples ;


  REAL *values=new REAL[num_features] ;
  REAL *vectors=new REAL[num_features*num_features] ;
  //  int fl ;
  //  symeigx(cov, num_features, values, vectors, num_features, &fl);

  if (0)
    {
      int lwork=4*num_features ;
      double *work=new double[lwork] ;
      int info ; char V='V', U='U' ;
      dsyev_(&V, &U, &num_features, cov, &num_features, values, work, &lwork, &info, 0) ;
    }

  CIO::message("done\nRunning matlab PCA code") ;
  cleaner_main(cov, num_features, 1e-5, &T, &num_dim) ;
  CIO::message("done\n") ;

  return true ;
}

/// initialize preprocessor from features
void CPCACut::cleanup()
{
  delete[] T ;
  T=NULL ;
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
  //  CIO::message("preprocessing vector of length %i to length %i\n", len, num_dim) ;

  REAL *ret=new REAL[num_dim] ;
  int onei=1 ;
  double zerod=0, oned=1 ;
  char N='N' ;
  dgemv_(&N, &num_dim, &len, &oned, T, &num_dim, f, &onei, &zerod, ret, &onei) ;

  len=num_dim ;
  return ret;
}

