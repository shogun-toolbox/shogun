#ifndef __GUIPREPROC_H__
#define __GUIPREPROC_H__

#include "preproc/PreProc.h"
#include "preproc/NormOne.h"
#include "preproc/PruneVarSubMean.h"
#include "preproc/PCACut.h"

class CGUI ;

class CGUIPreProc
{
 public:
  CGUIPreProc(CGUI*);
  ~CGUIPreProc();

  bool set_preproc(char* param);
  inline CPreProc * get_preproc() { return preproc ; }
 protected:
  CGUI* gui ;

  CPruneVarSubMean prunevarsubmean;
  CPCACut pcacut;
  CNormOne normone;
  CPreProc * preproc;
};
#endif
