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

