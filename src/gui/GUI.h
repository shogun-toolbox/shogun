#ifndef __GUI__H
#define __GUI__H

#include "guilib/GUIHMM.h"
#include "guilib/GUISVM.h"
#include "guilib/GUIPluginEstimate.h"
#include "guilib/GUIKNN.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUIPreProc.h"
#include "guilib/GUIFeatures.h"
#include "guilib/GUIMath.h"
#include "guilib/GUILabels.h"
#include "guilib/GUITime.h"

class CGUI
{
 public:

  CGUI(INT ac, char** av): guisvm(this), guihmm(this), guipluginestimate(this), guiknn(this), guikernel(this), 
    guipreproc(this), guifeatures(this), guilabels(this), guimath(this), guitime(this), argc(ac), argv(av) {} ;

  CGUISVM guisvm;
  CGUIHMM guihmm;
  CGUIPluginEstimate guipluginestimate;
  CGUIKNN guiknn;
  CGUIKernel guikernel;
  CGUIPreProc guipreproc;
  CGUIFeatures guifeatures;
  CGUILabels guilabels;
  CGUIMath guimath;
  CGUITime guitime;

  INT argc ;
  char** argv ;
} ;
#endif
