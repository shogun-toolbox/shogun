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
bool CRealFeatures::preproc_feature_matrix()
{
	if (preproc && !preprocessed)
		return (((CRealPreProc*) preproc)->apply_to_feature_matrix(this) != NULL);
	else
		return false;
}

bool CRealFeatures::load(FILE* src)
{
    assert(src!=NULL);

    unsigned char intlen=0;
    unsigned char doublelen=0;
    unsigned int endian=0;
    unsigned int fourcc=0;
    unsigned int preprocd=0;
    unsigned int num_vec=0;
    unsigned int num_feat=0;

    assert(fread(&intlen, sizeof(unsigned char), 1, src)==1);
    assert(fread(&doublelen, sizeof(unsigned char), 1, src)==1);
    assert(fread(&endian, (unsigned int) intlen, 1, src)== 1);
    assert(fread(&fourcc, (unsigned int) intlen, 1, src)==1);
    assert(fread(&num_vec, (unsigned int) intlen, 1, src)==1);
    assert(fread(&num_feat, (unsigned int) intlen, 1, src)==1);
    assert(fread(&preprocd, (unsigned int) intlen, 1, src)==1);
#error check for FOURCC , check for endianess+convert if not right+ more checks.

    delete[] feature_matrix;
    num_features=num_feat;
    num_vec=num_vectors;
    CIO::message("allocating feature matrix of size %.2fM\n", sizeof(double)*num_features*num_vectors/1024.0/1024.0);
    feature_matrix=new REAL[num_feat*num_vec];

    CIO::message("loading... be patient.\n");

    for (int i=0; i<num_vec; i++)
    {
	if (!(i % (num_vec/10+1)))
	    CIO::message("%02d%%.", (int) (100.0*i/num_vec));
	else if (!(i % (num_vec/200+1)))
	    CIO::message(".");

	assert(fread(&feature_matrix[num_features*i], doublelen, num_features, src)==num_features) ;
    }
    return true;
}

bool CRealFeatures::save(FILE* dest)
{
    unsigned char intlen=sizeof(unsigned int);
    unsigned char doublelen=sizeof(unsigned int);
    unsigned int endian=0x123456789;
    unsigned int fourcc='RFEA'; //id for real features
    unsigned int preprocd= (preprocessed) ? 1 : 0;
    unsigned int num_vec= (unsigned int) num_vectors;
    unsigned int num_feat= (unsigned int) num_features;

    assert(fwrite(&intlen, sizeof(unsigned char), 1, dest)==1);
    assert(fwrite(&doublelen, sizeof(double), 1, dest)==1);
    assert(fwrite(&endian, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&fourcc, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&num_vec, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&num_feat, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&preprocd, sizeof(unsigned int), 1, dest)==1);

    for (int i=0; i<num_vec; i++)
    {
	if (!(i % (num_vec/10+1)))
	    CIO::message("%02d%%.", (int) (100.0*i/num_vec));
	else if (!(i % (num_vec/200+1)))
	    CIO::message(".");

	long len;
       	bool free;
	double* f=get_feature_vector(i, len, free);
	assert(fwrite(f, sizeof(double), len, dest)==len) ;
	free_feature_vector(f, free) ;
    }
    
    return true;
}

