#include "Features.h"
#include "preproc/PreProc.h"
#include "lib/io.h"

#include <string.h>
#include <assert.h>

CFeatures::CFeatures(LONG size) : cache_size(size), preproc(NULL), num_preproc(0), preprocessed(NULL) 
{
	CIO::message("Feature object created (%ld)\n",this);
}

CFeatures::CFeatures(const CFeatures& orig) : preproc(orig.preproc), num_preproc(orig.num_preproc), preprocessed(orig.preprocessed)
{
	preprocessed=new bool[orig.num_preproc];
	assert(preprocessed);
	memcpy(preprocessed, orig.preprocessed, sizeof(bool)*orig.num_preproc);
}

CFeatures::CFeatures(CHAR* fname) : cache_size(0), preproc(NULL), num_preproc(0), preprocessed(false)
{
	load(fname);
	CIO::message("Feature object loaded (%ld)\n",this) ;
}

CFeatures::~CFeatures()
{
	CIO::message("Feature object destroyed (%ld)\n",this) ;
}

/// set preprocessor
INT CFeatures::add_preproc(CPreProc* p)
{ 
	CIO::message("%d preprocs currently, new preproc list is\n", num_preproc);
	INT i;

	bool* preprocd=new bool[num_preproc+1];
	CPreProc** pps=new CPreProc*[num_preproc+1];
	for (i=0; i<num_preproc; i++)
	{
		pps[i]=preproc[i];
		preprocd[i]=preprocessed[i];
	}
	delete[] preproc;
	delete[] preprocessed;
	preproc=pps;
	preprocessed=preprocd;
	preproc[num_preproc]=p;
	preprocessed[num_preproc]=false;

	num_preproc++;

	for (i=0; i<num_preproc; i++)
		CIO::message("preproc[%d]=%s %ld\n",i, preproc[i]->get_name(), preproc[i]) ;
	return num_preproc;
}

/// get current preprocessor
CPreProc* CFeatures::get_preproc(INT num)
{ 
	if (num<num_preproc)
		return preproc[num];
	else
		return NULL;
}

/// del current preprocessor
CPreProc* CFeatures::del_preproc(INT num)
{
	INT i,j;
	CPreProc** pps=NULL; 
	bool* preprocd=NULL; 
	CPreProc* removed_preproc=NULL;

	if (num_preproc>0)
		removed_preproc=preproc[num];

	if (num_preproc>1)
	{
		pps= new CPreProc*[num_preproc-1];
		preprocd= new bool[num_preproc-1];
	}

	if (pps && preprocd)
	{
		j=0;
		for (i=0; i<num_preproc; i++)
		{
			if (i!=num)
			{
				pps[j++]=preproc[i];
				preprocd[j++]=preprocessed[i];
			}
		}
		num_preproc--;
	}

	delete[] preproc;
	preproc=pps;

	for (i=0; i<num_preproc; i++)
		CIO::message("preproc[%d]=%s\n",i, preproc[i]->get_name()) ;

	return removed_preproc;
}

bool CFeatures::load(CHAR* fname)
{
	return false;
}

bool CFeatures::save(CHAR* fname)
{
	return false;
}
