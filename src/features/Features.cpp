#include "Features.h"
#include "preproc/PreProc.h"
#include "lib/io.h"

CFeatures::CFeatures(long size)
: cache_size(size), preproc(NULL), num_preproc(0), preprocessed(false) 
{
}

CFeatures::CFeatures(const CFeatures& orig)
: preproc(orig.preproc), preprocessed(orig.preprocessed)
{
}

CFeatures::~CFeatures()
{
  CIO::message("Feature object destroyed\n") ;
}

/// set preprocessor
int CFeatures::add_preproc(CPreProc* p)
{ 
	int i;
	CPreProc** pps=new CPreProc*[num_preproc+1];
	for (i=0; i<num_preproc; i++)
		pps[i]=preproc[i];
	preproc[num_preproc]=p;

	num_preproc++;

	for (i=0; i<num_preproc; i++)
		CIO::message("preproc[%d]=%s\n",i, preproc[i]->get_name()) ;
	return num_preproc;
}

/// set current preprocessor
CPreProc* CFeatures::get_preproc(int num)
{ 
	if (num<num_preproc)
		return preproc[num];
	else
		return NULL;
}

/// set current preprocessor
CPreProc* CFeatures::del_preproc(int num)
{
	int i,j;
	CPreProc** pps=NULL; 
	CPreProc* removed_preproc=NULL;

	if (num_preproc>0)
		removed_preproc=preproc[num];

	if (num_preproc>1)
		pps= new CPreProc*[num_preproc-1];

	if (pps)
	{
		j=0;
		for (i=0; i<num_preproc; i++)
		{
			if (i!=num)
				pps[j++]=preproc[i];
		}
		num_preproc--;
	}


	for (i=0; i<num_preproc; i++)
		CIO::message("preproc[%d]=%s\n",i, preproc[i]->get_name()) ;

	return removed_preproc;
}

int* CFeatures::get_labels(long &len)
{
  len=get_num_vectors() ;
  int* labels=new int[len] ;
  for (int i=0; i<len; i++)
    labels[i]=get_label(i) ;
  return labels ;
}

