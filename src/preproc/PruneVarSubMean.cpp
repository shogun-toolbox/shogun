#include "PruneVarSubMean.h"
#include "RealPreProc.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

#include <math.h>

CPruneVarSubMean::CPruneVarSubMean(bool divide)
  : CRealPreProc("PruneVarSubMean"), idx(NULL), mean(0), num_idx(0), divide_by_std(divide)
{
}

CPruneVarSubMean::~CPruneVarSubMean()
{
  delete[] idx ;
  delete[] mean;
}

/// initialize preprocessor from features
bool CPruneVarSubMean::init(CFeatures* f_)
{
    CIO::message("calling CPruneVarSubMean::init\n") ;

    CRealFeatures *f=(CRealFeatures*) f_ ;
    int num_examples=f->get_num_vectors() ;
    int num_features=((CRealFeatures*)f)->get_num_features() ;
   
    delete[] mean;
    delete[] idx;
    delete[] std; 
    mean=NULL;
    idx=NULL;
    std=NULL;

    mean=new double[num_features] ;
    double* var=new double[num_features] ;
    int i,j;

    for (i=0; i<num_features; i++)
    {
	mean[i]=0 ; var[i]=0 ;
    }

    // compute mean
    for (i=0; i<num_examples; i++)
    {
	long len ; bool free ;
	REAL* feature=f->get_feature_vector(i, len, free) ;

	for (int j=0; j<len; j++)
	    mean[j]+=feature[j] ;

	f->free_feature_vector(feature, free) ;
    }

    for (j=0; j<num_features; j++)
	mean[j]/=num_examples ;

    // compute var
    for (i=0; i<num_examples; i++)
    {
	long len ; bool free ;
	REAL* feature=f->get_feature_vector(i, len, free) ;

	for (int j=0; j<num_features; j++)
	    var[j]+=(mean[j]-feature[j])*(mean[j]-feature[j]) ;

	f->free_feature_vector(feature, free) ;
    }

    int num_ok=0;
    int* idx_ok=new int[num_features];

    for (j=0; j<num_features; j++)
    {
	var[j]/=num_examples ;

	if (var[j]>1e-6) 
	{
	    idx_ok[num_ok]=j ;
	    num_ok++ ;
	}
    }

    CIO::message("Reducing number of features from %i to %i\n", num_features, num_ok) ;

    delete[] idx ;
    idx=new int[num_ok];
    REAL* new_mean=new REAL[num_ok];
    std=new REAL[num_ok];

    for (j=0; j<num_ok; j++)
    {
	idx[j]=idx_ok[j] ;
	new_mean[j]=mean[idx_ok[j]];
	std[j]=sqrt(var[idx_ok[j]]);
    }
    num_idx=num_ok ;
    delete[] idx_ok ;
    delete[] mean;
    delete[] var;
    mean=new_mean;
    
    return true ;
}

/// initialize preprocessor from features
void CPruneVarSubMean::cleanup()
{
  delete[] idx;
  idx=NULL;
  delete[] mean;
  mean=NULL;
  delete[] std;
  std=NULL;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
REAL* CPruneVarSubMean::apply_to_feature_matrix(CFeatures* f)
{
    long num_vectors=0;
    long num_features=0;
    REAL* m=((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);

    CIO::message("get Feature matrix: %ix%i\n", num_vectors, num_features) ;
	CIO::message("Preprocessing feature matrix\n");
    for (long vec=0; vec<num_vectors; vec++)
	{
		REAL* v_src=&m[num_features*vec];
		REAL* v_dst=&m[num_idx*vec];

		if (divide_by_std)
		{
			for (long feat=0; feat<num_idx; feat++)
				v_dst[feat]=(v_src[idx[feat]]-mean[feat])/std[feat];
		}
		else
		{
			for (long feat=0; feat<num_idx; feat++)
				v_dst[feat]=(v_src[idx[feat]]-mean[feat]);
		}
	}
	
	((CRealFeatures*) f)->set_num_features(num_idx);
	((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);
	CIO::message("new Feature matrix: %ix%i\n", num_vectors, num_features);
    
    return m;
}

/// apply preproc on single feature vector
/// result in feature matrix
REAL* CPruneVarSubMean::apply_to_feature_vector(REAL* f, int &len)
{
  //CIO::message("preprocessing vector of length %i to length %i\n", len, num_idx) ;

  REAL *ret=new REAL[num_idx] ;
  if (divide_by_std)
  {
  for (int i=0; i<num_idx; i++)
    ret[i]=(f[idx[i]]-mean[i])/std[i];
  }
	else
  {
  for (int i=0; i<num_idx; i++)
    ret[i]=(f[idx[i]]-mean[i]);
  }

  len=num_idx ;
  return ret;
}
/// initialize preprocessor from file
bool CPruneVarSubMean::load_init_data(FILE* src)
{
	bool result=false;

    assert(fwrite(&idx, sizeof(int), 1, dest)==1) ;
    assert(fwrite(idx, sizeof(REAL), num_idx, dest)==num_idx) ;
    assert(fwrite(mean, sizeof(REAL), num_idx, dest)==num_idx) ;
    assert(fwrite(std, sizeof(REAL), num_idx, dest)==num_idx) ;
	return result;
}

/// save init-data (like transforamtion matrices etc) to file
bool CPruneVarSubMean::save_init_data(FILE* dst)
{
    assert(fwrite(&idx, sizeof(int), 1, dest)==1) ;
    assert(fwrite(idx, sizeof(REAL), num_idx, dest)==num_idx) ;
    assert(fwrite(mean, sizeof(REAL), num_idx, dest)==num_idx) ;
    assert(fwrite(std, sizeof(REAL), num_idx, dest)==num_idx) ;
	return false;
}
