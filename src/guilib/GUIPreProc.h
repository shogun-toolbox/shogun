#ifndef __GUIPREPROC_H__
#define __GUIPREPROC_H__

#include "preproc/PreProc.h"
#include "preproc/PruneVarSubMean.h"
#include "preproc/PCACut.h"

class CGUI ;

class CGUIPreProc
{
 public:
  CGUIPreProc(CGUI*);
  ~CGUIPreProc();

  CPreProc * get_preproc() { return preproc ; } ;
 protected:
  CGUI* gui ;

  CPreProc * preproc ;
  CPruneVarSubMean prunevarsubmean ;
  CPCACut pcacut ;
};
#endif
