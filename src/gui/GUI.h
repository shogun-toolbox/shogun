#ifndef __GUI__H
#define __GUI__H

#include "guilib/GUIHMM.h"
#include "guilib/GUISVM.h"
#include "guilib/GUIPluginEstimate.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUIPreProc.h"
#include "guilib/GUIFeatures.h"
#include "guilib/GUIMath.h"
#include "guilib/GUILabels.h"

class CGUI
{
 public:

  CGUI(INT ac, const CHAR**av): guisvm(this), guihmm(this), guipluginestimate(this), guikernel(this), 
    guipreproc(this), guifeatures(this), guilabels(this), argc(ac), argv(av) {} ;

  CGUISVM guisvm;
  CGUIHMM guihmm;
  CGUIPluginEstimate guipluginestimate;
  CGUIKernel guikernel;
  CGUIPreProc guipreproc;
  CGUIFeatures guifeatures;
  CGUILabels guilabels;
  CGUIMath guimath;

  INT argc ;
  const CHAR **argv ;
} ;

#endif
