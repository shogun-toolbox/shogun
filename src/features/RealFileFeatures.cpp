#include "features/RealFileFeatures.h"
#include "features/Features.h"
#include "preproc/RealPreProc.h"
#include "lib/io.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>

CRealFileFeatures::CRealFileFeatures(long size, char* fname) : CRealFeatures(size)
{
    working_file=fopen(fname, "r");
    working_filename=strdup(fname);
    assert(working_file!=NULL);
    intlen=0;
    doublelen=0;
    endian=0;
    fourcc=0;
    preprocd=0;
    labels=NULL;
    status=load_base_data();
}

CRealFileFeatures::CRealFileFeatures(long size, FILE* file) : CRealFeatures(size), working_file(file), working_filename(NULL)
{
    assert(working_file!=NULL);
    intlen=0;
    doublelen=0;
    endian=0;
    fourcc=0;
    preprocd=0;
    labels=NULL;
    status=load_base_data();
}

CRealFileFeatures::~CRealFileFeatures()
{
  delete[] feature_matrix;
  delete[] working_filename;
  delete[] labels;
}
  
CRealFileFeatures::CRealFileFeatures(const CRealFileFeatures & orig): CRealFeatures(orig), 
working_file(orig.working_file), status(orig.status)
{
    if (orig.working_filename)
	working_filename=strdup(orig.working_filename);
    if (orig.labels && get_num_vectors())
    {
	labels=new int[get_num_vectors()];
	memcpy(labels, orig.labels, sizeof(int)*get_num_vectors()); 
    }
}

CFeatures* CRealFileFeatures::duplicate() const
{
    return new CRealFileFeatures(*this);
}

REAL* CRealFileFeatures::compute_feature_vector(long num, long &len, REAL* target)
{
    assert(num<num_vectors);
    //CIO::message("f:%ld\n", num);
    len=num_features;
    REAL* featurevector=target;
	if (!featurevector)
	  featurevector=new REAL[num_features];
    assert(featurevector!=NULL);
    assert(working_file!=NULL);
    fseek(working_file, filepos+num_features*doublelen*num, SEEK_SET);
    assert(fread(featurevector, doublelen, num_features, working_file) == (unsigned long) num_features);
    return featurevector;
}

REAL* CRealFileFeatures::set_feature_matrix()
{
    assert(working_file!=NULL);
    fseek(working_file, filepos, SEEK_SET);
    delete[] feature_matrix;

    CIO::message("allocating feature matrix of size %.2fM\n", sizeof(double)*num_features*num_vectors/1024.0/1024.0);
    feature_matrix=new REAL[num_features*num_vectors];

    CIO::message("loading... be patient.\n");

    for (long i=0; i<(long) num_vectors; i++)
    {
	if (!(i % (num_vectors/10+1)))
	    CIO::message("%02d%%.", (int) (100.0*i/num_vectors));
	else if (!(i % (num_vectors/200+1)))
	    CIO::message(".");

	assert(fread(&feature_matrix[num_features*i], doublelen, num_features, working_file)== (unsigned long) num_features) ;
    }
	    CIO::message("done.\n");

    return feature_matrix;
}

int CRealFileFeatures::get_label(long idx)
{
    assert(idx<num_vectors);
    if (labels)
		return labels[idx];
    return 0;
}

bool CRealFileFeatures::load_base_data()
{
    assert(working_file!=NULL);
    unsigned int num_vec=0;
    unsigned int num_feat=0;

    assert(fread(&intlen, sizeof(unsigned char), 1, working_file)==1);
    assert(fread(&doublelen, sizeof(unsigned char), 1, working_file)==1);
    assert(fread(&endian, (unsigned int) intlen, 1, working_file)== 1);
    assert(fread(&fourcc, (unsigned int) intlen, 1, working_file)==1);
    assert(fread(&num_vec, (unsigned int) intlen, 1, working_file)==1);
    assert(fread(&num_feat, (unsigned int) intlen, 1, working_file)==1);
    assert(fread(&preprocd, (unsigned int) intlen, 1, working_file)==1);
    CIO::message("detected: intsize=%d, doublesize=%d, num_vec=%d, num_feat=%d, preprocd=%d\n", intlen, doublelen, num_vec, num_feat, preprocd);
#warning check for FOURCC , check for endianess+convert if not right+ more checks.
    filepos=ftell(working_file);
    set_num_vectors(num_vec);
    set_num_features(num_feat);
    preprocessed=preprocd==1;
	//CIO::message("seeking to: %ld\n", filepos+num_features*num_vectors*doublelen);
    fseek(working_file, filepos+num_features*num_vectors*doublelen, SEEK_SET);
    delete[] labels;
    labels= new int[num_vec];
    assert(labels!=NULL);
    assert(fread(labels, intlen, num_vec, working_file) == num_vec);
    return true;
}
