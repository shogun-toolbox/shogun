#include "lib/config.h"
#include "lib/Mathmatics.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>

#ifdef HAVE_ATLAS
extern "C" {
#include <cblas.h>
}

#ifdef HAVE_LAPACK
#include "lib/lapack.h"

#include "lib/common.h"
#include "PCACut.h"
#include "RealPreProc.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

CPCACut::CPCACut(INT do_whitening_, double thresh_) : CRealPreProc("PCACut", "PCAC"), T(NULL),
	num_dim(0), mean(NULL), initialized(false), do_whitening(do_whitening_), thresh(thresh_)
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
		assert(f->get_feature_class() == C_SIMPLE);
		assert(f->get_feature_type() == F_REAL);

		CIO::message(M_INFO,"calling CPCACut::init\n") ;
		INT num_vectors=((CRealFeatures*)f)->get_num_vectors() ;
		INT num_features=((CRealFeatures*)f)->get_num_features() ;
		CIO::message(M_INFO,"num_examples: %ld num_features: %ld \n", num_vectors, num_features);
		delete[] mean ;
		mean=new double[num_features+1] ;

		INT i,j;

		/// compute mean

		// clear
		for (j=0; j<num_features; j++)
		{
			mean[j]=0 ; 
		}

		// sum 
		for (i=0; i<num_vectors; i++)
		{
			INT len;
			bool free;
			REAL* vec=((CRealFeatures*) f)->get_feature_vector(i, len, free);
			for (j=0; j<num_features; j++)
			{
				mean[j]+= vec[j];
			}
			((CRealFeatures*) f)->free_feature_vector(vec, i, free);
		}

		//divide
		for (j=0; j<num_features; j++)
			mean[j]/=num_vectors;

		CIO::message(M_INFO,"done.\nComputing covariance matrix... of size %.2f M\n", num_features*num_features/1024.0/1024.0) ;
		double *cov=new double[num_features*num_features] ;
		assert(cov!=NULL) ;

		for (j=0; j<num_features*num_features; j++)
			cov[j]=0.0 ;

		for (i=0; i<num_vectors; i++)
		{
			if (!(i % (num_vectors/10+1)))
				CIO::progress(i, 0, num_vectors);

			INT len;
			bool free;

			REAL* vec=((CRealFeatures*) f)->get_feature_vector(i, len, free) ;

			for (INT j=0; j<num_features; j++)
				vec[j]-=mean[j] ;

			/// A = 1.0*xy^T+A blas
			cblas_dger(CblasColMajor, num_features,num_features, 1.0, vec, 1, 
				 vec, 1, cov, (int)num_features) ;

			//for (INT k=0; k<num_features; k++)
			//	for (INT l=0; l<num_features; l++)
			//          cov[k*num_features+l]+=feature[l]*feature[k] ;

			((CRealFeatures*) f)->free_feature_vector(vec, i, free) ;
		}

		CIO::message(M_MESSAGEONLY, "done.           \n");

		for (i=0; i<num_features; i++)
			for (j=0; j<num_features; j++)
				cov[i*num_features+j]/=num_vectors ;

		CIO::message(M_INFO,"done\n") ;

		CIO::message(M_INFO,"Computing Eigenvalues ... ") ;
		INT lwork=3*num_features ;
		double* work=new double[lwork] ;
		double* eigenvalues=new double[num_features] ;
		INT info;
		CHAR V='V';
		CHAR U='U';
		INT ord= (int) num_features;
		INT lda= (int) num_features;

		for (i=0; i<num_features; i++)
			eigenvalues[i]=0;

		// lapack sym matrix eigenvalues+vectors
		dsyev_(&V, &U, &ord, cov, &lda, eigenvalues, work, &lwork, &info) ;
		delete[] work;

		num_dim=0;
		for (i=0; i<num_features; i++)
		{
			//	  CIO::message("EV[%i]=%e\n", i, values[i]) ;
			if (eigenvalues[i]>thresh)
				num_dim++ ;
		} ;

		CIO::message(M_INFO,"Done\nReducing from %i to %i features..", num_features, num_dim) ;

		delete[] T;
		T=new REAL[num_dim*num_features] ;
		num_old_dim=num_features;

		assert(T!=NULL) ;
		if (do_whitening)
		{
			INT offs=0 ;
			for (i=0; i<num_features; i++)
			{
				if (eigenvalues[i]>1e-6)
				{
					for (INT j=0; j<num_features; j++)
						T[offs+j*num_dim]=cov[num_features*i+j]/sqrt(eigenvalues[i]) ;
					offs++ ;
				} ;
			}
		} ;

		delete[] eigenvalues;
		delete[] cov;
		initialized=true;
		CIO::message(M_INFO,"Done\n") ;
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
	INT num_vectors=0;
	INT num_features=0;

	REAL* m=((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);
	CIO::message(M_INFO,"get Feature matrix: %ix%i\n", num_vectors, num_features) ;

	if (m)
	{
		CIO::message(M_INFO,"Preprocessing feature matrix\n");
		REAL* res= new REAL[num_dim];
		double* sub_mean= new double[num_features];

		for (INT vec=0; vec<num_vectors; vec++)
		{
			INT i;

			for (i=0; i<num_features; i++)
				sub_mean[i]=m[num_features*vec+i]-mean[i] ;

			cblas_dgemv(CblasColMajor, CblasNoTrans, num_dim, num_features, 1.0,
				  T, num_dim, sub_mean, 1, 0, res, 1); 

			REAL* m_transformed=&m[num_dim*vec];
			for (i=0; i<num_dim; i++)
				m_transformed[i]=m[i];
		}
		delete[] res;
		delete[] sub_mean;

		((CRealFeatures*) f)->set_num_features(num_dim);
		((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);
		CIO::message(M_INFO,"new Feature matrix: %ix%i\n", num_vectors, num_features);
	}

	return m;
}

/// apply preproc on single feature vector
/// result in feature matrix
REAL* CPCACut::apply_to_feature_vector(REAL* f, INT &len)
{
	REAL *ret=new REAL[num_dim];
	REAL *sub_mean=new REAL[len];
	for (INT i=0; i<len; i++)
		sub_mean[i]=f[i]-mean[i];

	cblas_dgemv(CblasColMajor, CblasNoTrans, num_dim, len, 1.0 , T, num_dim, sub_mean, 1, 0, ret, 1) ;
//void cblas_dgemv(const enum CBLAS_ORDER order,
//                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
//                 const double alpha, const double *A, const int lda,
//                 const double *X, const int incX, const double beta,
//                 double *Y, const int incY);
//

	delete[] sub_mean ;
	len=num_dim ;
	//	  CIO::message("num_dim: %d\n", num_dim);
	return ret;
}

/// initialize preprocessor from file
bool CPCACut::load_init_data(FILE* src)
{
	assert(fread(&num_dim, sizeof(int), 1, src)==1);
	assert(fread(&num_old_dim, sizeof(int), 1, src)==1);
	delete[] mean;
	delete[] T;
	mean=new double[num_dim];
	T=new double[num_dim*num_old_dim];
	assert (mean!=NULL && T!=NULL);
	assert(fread(mean, sizeof(double), num_old_dim, src)==(UINT) num_old_dim);
	assert(fread(T, sizeof(double), num_dim*num_old_dim, src)==(UINT) num_old_dim*num_dim);
	return true;
}

/// save init-data (like transforamtion matrices etc) to file
bool CPCACut::save_init_data(FILE* dst)
{
	assert(fwrite(&num_dim, sizeof(int), 1, dst)==1);
	assert(fwrite(&num_old_dim, sizeof(int), 1, dst)==1);
	assert(fwrite(mean, sizeof(double), num_old_dim, dst)==(UINT) num_old_dim);
	assert(fwrite(T, sizeof(double), num_dim*num_old_dim, dst)==(UINT) num_old_dim*num_dim);
	return true;
}

#endif // HAVE_LAPACK
#endif // HAVE_ATLAS
