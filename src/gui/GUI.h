#ifndef __GUI__H
#define __GUI__H

#include "guilib/GUIHMM.h"
#include "guilib/GUISVM.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUIObservation.h"
#include "guilib/GUIPreProc.h"
#include "guilib/GUIFeatures.h"
#include "guilib/GUIMath.h"

class CGUI
{
 public:

  CGUI(int ac, const char**av): guisvm(this), guihmm(this), guikernel(this), 
    guiobs(this), guipreproc(this), guifeatures(this), argc(ac), argv(av) {} ;

  CGUISVM guisvm;
  CGUIHMM guihmm;
  CGUIKernel guikernel;
  CGUIObservation guiobs;
  CGUIPreProc guipreproc;
  CGUIFeatures guifeatures;
  CGUIMath guimath;

  int argc ;
  const char **argv ;
} ;

#endif
