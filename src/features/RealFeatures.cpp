#include <assert.h>
#include "features/RealFeatures.h"
#include "preproc/RealPreProc.h"
#include <string.h>
#include "lib/io.h"

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
	memcpy(feature_matrix, orig.feature_matrix, sizeof(double)*num_vectors*num_features); 
    }
}

/// get feature vector for sample num
REAL* CRealFeatures::get_feature_vector(long num, long &len, bool &free)
{
  len=num_features; 
  assert(num<num_vectors);
  
  if (feature_matrix)
    {
//      CIO::message("returning %i th column of feature_matrix\n", (int)num) ;
      free=false ;
      return &feature_matrix[num*num_features];
    } 
  else
    {
      //CIO::message("computing %i th feature vector\n", (int)num) ;
      free=true ;
      REAL* feat=compute_feature_vector(num, len) ;
      if (preproc)
	{
	  //CIO::message("preprocessing %i th feature vector\n", (int)num) ;
	  int len2=len ;
	  REAL* feat2 = ((CRealPreProc*) preproc)->apply_to_feature_vector(feat, len2);
	  delete[] feat ;
	  CIO::message("len2: %d len: %d\n", len2, len);
	  len=num_features=len2 ;
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
bool CRealFeatures::preproc_feature_matrix(bool force_preprocessing)
{
	CIO::message("preproc: %d, preprocd: %d, force: %d\n", preproc, preprocessed, force_preprocessing);

	if ( preproc && (!preprocessed || force_preprocessing) )
	{
	    preprocessed=true;	
	    return (((CRealPreProc*) preproc)->apply_to_feature_matrix(this) != NULL);
	}
	else
	{
		CIO::message("no feature matrix available or features already preprocessed - skipping.\n");
		return false;
	}
}

bool CRealFeatures::save(FILE* dest)
{
    int i;
	long len;
	bool free;
#warning num_features must not correspond with the length of a feature vectore since that one might be preprocessed
	double* f=get_feature_vector(0, len, free);
	free_feature_vector(f, free) ;

    unsigned char intlen=sizeof(unsigned int);
    unsigned char doublelen=sizeof(double);
    unsigned int endian=0x12345678;
    unsigned int fourcc='RFEA'; //id for real features
    unsigned int preprocd= (preprocessed) ? 1 : 0;
    unsigned int num_vec= (unsigned int) num_vectors;
    unsigned int num_feat= (unsigned int) len; // this is bit of a hack - suggestions please !  //num_features;

	  CIO::message("saving matrix of size %dx%d\n", num_vec,num_feat) ;
    assert(fwrite(&intlen, sizeof(unsigned char), 1, dest)==1);
    assert(fwrite(&doublelen, sizeof(unsigned char), 1, dest)==1);
    assert(fwrite(&endian, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&fourcc, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&num_vec, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&num_feat, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&preprocd, sizeof(unsigned int), 1, dest)==1);

    for (i=0; i<num_vec; i++)
    {
	if (!(i % (num_vec/10+1)))
	    CIO::message("%02d%%.", (int) (100.0*i/num_vec));
	else if (!(i % (num_vec/200+1)))
	    CIO::message(".");

	f=get_feature_vector(i, len, free);
	assert(fwrite(f, sizeof(double), len, dest)==len) ;
	free_feature_vector(f, free) ;
    }

    long num_lab=0;
    int* labels=get_labels(num_lab);
    assert(num_lab==num_vec);
    assert(fwrite(labels, sizeof(int), num_vec, dest)==num_vec) ;
    
    return true;
}

