#include "lib/common.h"
#include "PCACut.h"
#include "RealPreProc.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"
#include <math.h>
//#include <libmmfile.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

extern "C" void cleaner_main(double *covZ, int dim, double thresh,
			     double **T, int *num_dim)  ;

CPCACut::CPCACut()
  : CRealPreProc("PCACut"), T(NULL), num_dim(0), mean(NULL), initialized(false)
{
}

CPCACut::~CPCACut()
{
  delete[] T;
  delete[] mean;
}

/// initialize preprocessor from features
bool CPCACut::init(CFeatures* f)
{
    if (!initialized)
    {
	CIO::message("calling CPCACut::init\n") ;
	int num_vectors=((CRealFeatures*)f)->get_number_of_examples() ;
	int num_features=((CRealFeatures*)f)->get_num_features() ;
	CIO::message("num_examples: %ld num_features: %ld \n", num_vectors, num_features);
	delete[] mean ;
	mean=new double[num_features+1] ;

	int i,j;

	/// compute mean

	// clear
	for (j=0; j<num_features; j++)
	{
	    mean[j]=0 ; 
	}

	// sum 
	for (i=0; i<num_vectors; i++)
	{
	    long len;
	    bool free;
	    REAL* vec=((CRealFeatures*) f)->get_feature_vector(i, len, free);
	    for (j=0; j<num_features; j++)
	    {
		mean[j]+= vec[j];
	    }
	    ((CRealFeatures*) f)->free_feature_vector(vec, free);
	}

	//divide
	for (j=0; j<num_features; j++)
	    mean[j]/=num_vectors;

	CIO::message("done.\nComputing covariance matrix... of size %.2f M\n", num_features*num_features/1024.0/1024.0) ;
	double *cov=new double[num_features*num_features] ;
	assert(cov!=NULL) ;

	for (j=0; j<num_features*num_features; j++)
	    cov[j]=0.0 ;

	for (i=0; i<num_vectors; i++)
	{
	    if (!(i % (num_vectors/10+1)))
		CIO::message("%02d%%.", (int) (100.0*i/num_vectors));
	    else if (!(i % (num_vectors/200+1)))
		CIO::message(".");

	    long len;
	    bool free;

	    REAL* vec=((CRealFeatures*) f)->get_feature_vector(i, len, free) ;

	    for (int j=0; j<num_features; j++)
		vec[j]-=mean[j] ;

	    double oned=1.0;
	    int onei=1;
	    int lda=(int) num_features;

	    /// A = 1.0*xy^T+A blas
	    dger_(&num_features,&num_features, &oned, vec, &onei, vec, &onei, cov, &lda) ;

	    //for (int k=0; k<num_features; k++)
	    //	for (int l=0; l<num_features; l++)
	    //          cov[k*num_features+l]+=feature[l]*feature[k] ;

	    ((CRealFeatures*) f)->free_feature_vector(vec, free) ;
	}

	for (i=0; i<num_features; i++)
	    for (j=0; j<num_features; j++)
		cov[i*num_features+j]/=num_vectors ;

	CIO::message("done\n") ;

	CIO::message("Computing Eigenvalues ... ") ;
	int lwork=3*num_features ;
	double* work=new double[lwork] ;
	double* eigenvalues=new double[num_features] ;
	int info;
	char V='V';
	char U='U';
	int ord= (int) num_features;
	int lda= (int) num_features;

	for (i=0; i<num_features; i++)
	    eigenvalues[i]=0;

	// lapack sym matrix eigenvalues+vectors
	dsyev_(&V, &U, &ord, cov, &lda, eigenvalues, work, &lwork, &info) ;
	delete[] work;

	num_dim=0;
	for (i=0; i<num_features; i++)
	{
	    //	  CIO::message("EV[%i]=%e\n", i, values[i]) ;
	    if (eigenvalues[i]>1e-6)
		num_dim++ ;
	} ;

	CIO::message("Done\nReducing from %i to %i features..", num_features, num_dim) ;

	delete[] T;
	T=new REAL[num_dim*num_features] ;
	assert(T!=NULL) ;
	int offs=0 ;
	for (i=0; i<num_features; i++)
	{
	    if (eigenvalues[i]>1e-6)
	    {
		for (int j=0; j<num_features; j++)
		    T[offs+j*num_dim]=cov[num_features*i+j]/sqrt(eigenvalues[i]) ;
		offs++ ;
	    } ;
	}

	delete[] eigenvalues;
	delete[] cov;
	initialized=true;
	CIO::message("Done\n") ;
	return true ;
    }
    return 
	false;
}

/// initialize preprocessor from features
void CPCACut::cleanup()
{
  delete[] T ;
  T=NULL ;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
REAL* CPCACut::apply_to_feature_matrix(CFeatures* f)
{
    long num_vectors=0;
    long num_features=0;

    REAL* m=((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);
    CIO::message("get Feature matrix: %ix%i\n", num_vectors, num_features) ;

    if (m)
    {
	CIO::message("Preprocessing feature matrix\n");
	REAL* res= new REAL[num_dim];
	double* sub_mean= new double[num_features];

	for (int vec=0; vec<num_vectors; vec++)
	{
	    int i;

	    for (i=0; i<num_features; i++)
		sub_mean[i]=m[num_features*vec+i]-mean[i] ;

	    int onei=1;
	    double zerod=0;
	    double oned=1;
	    char N='N';
	    int num_f=num_features;
	    int num_d=num_dim;
	    int lda=num_dim;

	    CIO::message("dgemv args: num_f: %d, num_d: %d\n", num_f, num_d) ;
#warning num_f might be num_features-1 or it is a bug in the SUN performance library (might apply to num_d too)
	    dgemv_(&N, &num_d, &num_f, &oned, T, &lda, sub_mean, &onei, &zerod, res, &onei); 

	    REAL* m_transformed=&m[num_dim*vec];
	    for (i=0; i<num_dim; i++)
		m_transformed[i]=m[i];
	}
	delete[] res;
	delete[] sub_mean;

	((CRealFeatures*) f)->set_num_features(num_dim);
	((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);
	CIO::message("new Feature matrix: %ix%i\n", num_vectors, num_features);
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
