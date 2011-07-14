/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
 
%define REGRESSION_DOCSTR
"The `Regression` module gathers all regression methods available in the SHOGUN toolkit."
%enddef

%module(docstring=REGRESSION_DOCSTR) Regression
#undef DOCSTR

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Regression_doxygen.i"
#endif
#endif

/* Include Module Definitions */
%include "SGBase.i"
%include "Features_includes.i"
%include "Kernel_includes.i"
%include "Distance_includes.i"
%include "Classifier_includes.i"
%include "Regression_includes.i"
%include "Preprocessor_includes.i"
%include "Library_includes.i"
%include "Distribution_includes.i"


%import "Features.i"
%import "Kernel.i"
%import "Distance.i"
%import "Classifier.i"

/* Remove C Prefix */
%rename(BaseRegression) CRegression;
/*%rename(Machine) CMachine;
%rename(KernelMachine) CKernelMachine; */
%rename(KRR) CKRR;
%rename(LibSVR) CLibSVR;
%rename(MKL) CMKL;
%rename(MKLRegression) CMKLRegression;
#ifdef USE_SVMLIGHT
%rename(SVRLight) CSVRLight;
#endif //USE_SVMLIGHT

/* Include Class Headers to make them visible from within the target language */
%include <shogun/regression/Regression.h>
/*%include <shogun/machine/Machine.h>
%include <shogun/machine/KernelMachine.h>*/
%include <shogun/regression/KRR.h>
%include <shogun/regression/svr/LibSVR.h>
%include <shogun/classifier/mkl/MKL.h>
%include <shogun/regression/svr/MKLRegression.h>


#ifdef USE_SVMLIGHT
%include <shogun/regression/svr/SVRLight.h>
#endif //USE_SVMLIGHT
