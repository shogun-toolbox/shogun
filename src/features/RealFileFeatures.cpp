#include "features/RealFileFeatures.h"
#include "features/Features.h"
#include "preproc/RealPreProc.h"
#include "lib/io.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>

CRealFileFeatures::CRealFileFeatures(LONG size, CHAR* fname) : CRealFeatures(size)
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

CRealFileFeatures::CRealFileFeatures(LONG size, FILE* file) : CRealFeatures(size), working_file(file), working_filename(NULL)
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

REAL* CRealFileFeatures::compute_feature_vector(INT num, INT &len, REAL* target)
{
    assert(num<num_vectors);
    len=num_features;
    REAL* featurevector=target;
	if (!featurevector)
	  featurevector=new REAL[num_features];
    assert(featurevector!=NULL);
    assert(working_file!=NULL);
    fseek(working_file, filepos+num_features*doublelen*num, SEEK_SET);
    assert(fread(featurevector, doublelen, num_features, working_file) == (size_t) num_features);
    return featurevector;
}

REAL* CRealFileFeatures::load_feature_matrix()
{
    assert(working_file!=NULL);
    fseek(working_file, filepos, SEEK_SET);
    delete[] feature_matrix;

    CIO::message(M_INFO, "allocating feature matrix of size %.2fM\n", sizeof(double)*num_features*num_vectors/1024.0/1024.0);
    feature_matrix=new REAL[num_features*num_vectors];

    CIO::message(M_INFO, "loading... be patient.\n");

    for (INT i=0; i<(INT) num_vectors; i++)
    {
	if (!(i % (num_vectors/10+1)))
	    CIO::message(M_MESSAGEONLY, "%02d%%.", (int) (100.0*i/num_vectors));
	else if (!(i % (num_vectors/200+1)))
	    CIO::message(M_MESSAGEONLY, ".");

	assert(fread(&feature_matrix[num_features*i], doublelen, num_features, working_file)== (size_t) num_features) ;
    }
	    CIO::message(M_INFO, "done.\n");

    return feature_matrix;
}

INT CRealFileFeatures::get_label(INT idx)
{
    assert(idx<num_vectors);
    if (labels)
		return labels[idx];
    return 0;
}

bool CRealFileFeatures::load_base_data()
{
    assert(working_file!=NULL);
    UINT num_vec=0;
    UINT num_feat=0;

    assert(fread(&intlen, sizeof(BYTE), 1, working_file)==1);
    assert(fread(&doublelen, sizeof(BYTE), 1, working_file)==1);
    assert(fread(&endian, (UINT) intlen, 1, working_file)== 1);
    assert(fread(&fourcc, (UINT) intlen, 1, working_file)==1);
    assert(fread(&num_vec, (UINT) intlen, 1, working_file)==1);
    assert(fread(&num_feat, (UINT) intlen, 1, working_file)==1);
    assert(fread(&preprocd, (UINT) intlen, 1, working_file)==1);
    CIO::message(M_INFO, "detected: intsize=%d, doublesize=%d, num_vec=%d, num_feat=%d, preprocd=%d\n", intlen, doublelen, num_vec, num_feat, preprocd);
    filepos=ftell(working_file);
    set_num_vectors(num_vec);
    set_num_features(num_feat);
    fseek(working_file, filepos+num_features*num_vectors*doublelen, SEEK_SET);
    delete[] labels;
    labels= new int[num_vec];
    assert(labels!=NULL);
    assert(fread(labels, intlen, num_vec, working_file) == num_vec);
    return true;
}
