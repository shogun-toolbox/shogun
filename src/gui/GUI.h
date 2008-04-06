/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUI__H
#define __GUI__H

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "base/SGObject.h"

#include "guilib/GUIHMM.h"
#include "guilib/GUIClassifier.h"
#include "guilib/GUIPluginEstimate.h"
#include "guilib/GUIKNN.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUIPreProc.h"
#include "guilib/GUIFeatures.h"
#include "guilib/GUIMath.h"
#include "guilib/GUILabels.h"
#include "guilib/GUITime.h"

#include "guilib/GUIDistance.h"

class CGUI : public CSGObject
{
 public:

  CGUI(INT ac, char** av): guiclassifier(this), guihmm(this), guipluginestimate(this), guiknn(this), guikernel(this), 
    guipreproc(this), guifeatures(this), guilabels(this), guimath(this), guitime(this), guidistance(this) ,argc(ac), argv(av) {} ;

  CGUIClassifier guiclassifier;
  CGUIHMM guihmm;
  CGUIPluginEstimate guipluginestimate;
  CGUIKNN guiknn;
  CGUIKernel guikernel;
  CGUIPreProc guipreproc;
  CGUIFeatures guifeatures;
  CGUILabels guilabels;
  CGUIMath guimath;
  CGUITime guitime;
	
  CGUIDistance guidistance;

  INT argc ;
  char** argv ;
} ;
#endif //HAVE_SWIG
#endif //__GUI__H
