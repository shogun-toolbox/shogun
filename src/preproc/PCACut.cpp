#include "PCACut.h"
#include "RealPreProc.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"
#include <math.h>
//#include <libmmfile.h>


extern "C" void cleaner_main(double *covZ, int dim, double thresh,
			     double **T, int *num_dim)  ;

CPCACut::CPCACut()
  : CRealPreProc("PCACut"), T(NULL), num_dim(0), mean(NULL) 
{
}

CPCACut::~CPCACut()
{
  delete[] T ;
  delete[] mean ;
}

/// initialize preprocessor from features
bool CPCACut::init(CFeatures* f_)
{
  CIO::message("calling CPCACut::init\n") ;
 
  CRealFeatures *f=(CRealFeatures*) f_ ;
  int num_examples=f->get_number_of_examples() ;
  int num_features=((CRealFeatures*)f)->get_num_features() ;
  delete[] mean ;
  mean=new double[num_features] ;
  double *feature=NULL ;
  int j ;
  for (j=0; j<num_features; j++)
      mean[j]=0 ; 

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
  for (int j=0; j<num_features*num_features; j++)
    cov[j]=0.0 ;

  for (int i=0; i<num_examples; i++)
    {
      long len ; bool free ;
      feature=f->get_feature_vector(i, len, free) ;

      for (int j=0; j<num_features; j++)
	feature[j]-=mean[j] ;

      double oned=1.0 ; int onei=1 ;
      //      dger_(&num_features,&num_features, &oned, feature, &onei, feature, &onei, cov, &num_features) ;

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

//    {
//      CIO::message("done\nRunning matlab PCA code ") ;
//      libmmfileInitialize() ;
//      cleaner_main(cov, num_features, 1e-4, &T, &num_dim) ;
//      libmmfileTerminate() ;
//      CIO::message("done\n") ;
//      for (int k=0; k<num_features; k++)
//        {
//  	for (int l=0; l< num_dim; l++)
//  	  CIO::message("%e ", T[k*num_dim+l]) ;
//  	CIO::message("\n") ;
//        } ;
//    }
    {
      int lwork=4*num_features ;
      double *work=new double[lwork] ;
      int info ; char V='V', U='U' ;
      dsyev_(&V, &U, &num_features, cov, &num_features, values, work, &lwork, &info) ;
      int num_ok=0 ;
      for (int i=0; i<num_features; i++)
	{
	  CIO::message("EV[%i]=%e\n", i, values[i]) ;
	  if (values[i]>1e-4)
	    num_ok++ ;
	} ;
//        for (int k=0; k<num_features; k++)
//  	{
//  	  for (int l=0; l< num_features; l++)
//  	    CIO::message("%e ", cov[k*num_features+l]) ;
//  	  CIO::message("\n") ;
//  	} ;
      T=new REAL[num_ok*num_features] ;
      int num_ok2=0 ;
      num_dim=num_ok ;
      for (int i=0; i<num_features; i++)
	{
	  if (values[i]>1e-4)
	    {
	      for (int j=0; j<num_features; j++)
		T[num_ok2+j*num_ok]=cov[num_features*i+j]/sqrt(values[i]) ;
	      num_ok2++ ;
	    } ;
	} ;
//  	  CIO::message("\n") ;
//        for (int k=0; k<num_features; k++)
//  	{
//  	  for (int l=0; l< num_dim; l++)
//  	    CIO::message("%e ", T[l+k*num_dim]) ;
//  	  CIO::message("\n") ;
//  	} ;
//  	  CIO::message("\n") ;
//        for (int k=0; k<num_features; k++)
//  	{
//  	  for (int l=0; l< num_dim; l++)
//  	    CIO::message("%e ", T2[l+k*num_dim]) ;
//  	  CIO::message("\n") ;
//  	} ;
      
    }


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
#warning crashes right here!
  long num_vectors=0;
  long num_features=0;
  
  REAL* m=((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);
  CIO::message("get Feature matrix: %ix%i\n", num_vectors, num_features) ;
  
  if (m)
    {
      REAL* res= new REAL[num_dim];
      REAL* sub_mean= new REAL[num_features];
      for (int vec=0; vec<num_vectors; vec++)
	{
	  int onei=1 ;
	  double zerod=0, oned=1;
	  char N='N';
	  int i;
	  for (i=0; i<num_features; i++)
	    sub_mean[i]=m[num_features*vec+i]-mean[i] ;
	  
	  int num_feat=num_features;
	  dgemv_(&N, &num_dim, &num_feat, &oned, T, &num_dim, sub_mean, &onei, &zerod, res, &onei) ;
	  
	  REAL* m_transformed=&m[num_dim*vec];
	  for (i=0; i<num_dim; i++)
	    m_transformed[i]=m[i];
	}
      delete[] res;
      delete[] sub_mean;
      
      ((CRealFeatures*) f)->set_num_features(num_dim);
    }
  
  return m;
}

/// apply preproc on single feature vector
/// result in feature matrix
REAL* CPCACut::apply_to_feature_vector(REAL* f, int &len)
{
  REAL *ret=new REAL[num_dim] ;
  int onei=1 ;
  double zerod=0, oned=1 ;
  char N='N' ;
  REAL *sub_mean=new REAL[len] ;
  for (int i=0; i<len; i++)
    sub_mean[i]=f[i]-mean[i] ;
  
  dgemv_(&N, &num_dim, &len, &oned, T, &num_dim, sub_mean, &onei, &zerod, ret, &onei) ;

  delete[] sub_mean ;
  len=num_dim ;
  return ret;
}

