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

bool CFeatures::preproc_feature_matrix()
{
	if (preproc)
		preproc->apply_to_feature_matrix(this);
}

int* CFeatures::get_labels(long &len)
{
  len=get_number_of_examples() ;
  int* labels=new int[len] ;
  for (int i=0; i<len; i++)
    labels[i]=get_label(i) ;
  return labels ;
}

