#include "Features.h"

CFeatures::CFeatures()
: preproc(NULL)
{
    
}

CFeatures::~CFeatures()
{
  delete preproc;
}

/// set preprocessor
void CFeatures::set_preproc(CPreProc* p)
{ 
  delete preproc; 
  preproc=p ;
}

/// set current preprocessor
CPreProc* CFeatures::get_preproc()
{ 
  return preproc;
}

int* CFeatures::get_labels(int idx, int &len)
{
  len=get_number_of_examples() ;
  int* labels=new int[len] ;
  for (int i=0; i<len; i++)
    labels[i]=get_label(idx) ;
  return labels ;
} ;
