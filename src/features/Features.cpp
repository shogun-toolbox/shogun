#include "Features.h"
#include "preproc/PreProc.h"
#include "lib/io.h"

CFeatures::CFeatures(long size)
: cache_size(size), preproc(NULL), preprocessed(false)
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
void CFeatures::set_preproc(CPreProc* p)
{ 
  preproc=p;
  CIO::message("set preproc %ld\n", p) ;
}

/// set current preprocessor
CPreProc* CFeatures::get_preproc()
{ 
  return preproc;
}

int* CFeatures::get_labels(long &len)
{
  len=get_num_vectors() ;
  int* labels=new int[len] ;
  for (int i=0; i<len; i++)
    labels[i]=get_label(i) ;
  return labels ;
}

