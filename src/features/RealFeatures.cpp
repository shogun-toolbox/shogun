#include "features/RealFeatures.h"
#include "lib/File.h"

CFeatures* CRealFeatures::duplicate() const
{
	return new CRealFeatures(*this);
}

bool CRealFeatures::load(char* fname)
{
	bool status=false;
	num_vectors=1;
    num_features=0;
	CFile f(fname, 'r', F_REAL);
	feature_matrix=f.load_real_data(NULL, num_features);

    if (!f.is_ok())
		CIO::message("loading file \"%s\" failed", fname);
	else
		status=true;

	return status;
}

bool CRealFeatures::save(char* fname)
{
    long i;
	long len;
	bool free;
#warning num_features must not correspond with the length of a feature vectore since that one might be preprocessed
	double* f=get_feature_vector(0, len, free);
	free_feature_vector(f, 0, free) ;

	FILE* dest=fopen(fname,"w");

	assert(dest);
    unsigned char intlen=sizeof(unsigned int);
    unsigned char doublelen=sizeof(double);
    unsigned int endian=0x12345678;
    unsigned int fourcc='RFEA'; //id for ST features
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

    for (i=0; i< (long) num_vec; i++)
    {
	if (!(i % (num_vec/10+1)))
	    CIO::message("%02d%%.", (int) (100.0*i/num_vec));
	else if (!(i % (num_vec/200+1)))
	    CIO::message(".");

	f=get_feature_vector(i, len, free);
	assert(((long)fwrite(f, (long) sizeof(double), len, dest))==len) ;
	free_feature_vector(f, i, free) ;
    }

    //long num_lab=0;
    //int* labels=get_labels(num_lab);
    //assert(num_lab==(long) num_vec);
    //assert(fwrite(labels, sizeof(int), num_vec, dest)==num_vec) ;
    
	if (dest)
		fclose(dest);
    return true;
}
