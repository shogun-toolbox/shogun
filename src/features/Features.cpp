#include "Features.h"
#include "preproc/PreProc.h"
#include "lib/io.h"

CFeatures::CFeatures()
: preproc(NULL)
{
}

CFeatures::~CFeatures()
{
  //delete preproc;
  CIO::message("Feature object destroyed\n") ;
}

/// set preprocessor
void CFeatures::set_preproc(CPreProc* p)
{ 
  //  if (preproc)
  //    delete preproc; 
  preproc=p ;
  CIO::message("set preproc %ld\n", p) ;
}

/// set current preprocessor
CPreProc* CFeatures::get_preproc()
{ 
  return preproc;
}

int* CFeatures::get_labels(long &len)
{
  len=get_number_of_examples() ;
  int* labels=new int[len] ;
  for (int i=0; i<len; i++)
    labels[i]=get_label(i) ;
  return labels ;
}

