/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "Evaluation_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%feature("autodoc", "get_ROC(self) -> numpy array of Receiver Operating Curve points (2 floats)") get_ROC;
%feature("autodoc", "get_all_CC(self) -> numpy array of Cross Correlation coefficients (2 floats)") get_all_CC;
%feature("autodoc", "get_all_WRAcc(self) -> numpy array of Weighted Relative Accuracy values (2 floats)") get_all_WRAcc;
%feature("autodoc", "get_all_BAL(self) -> numpy array of Balanced Error values (2 floats)") get_BAL;
%feature("autodoc", "get_accuracyROC(self) -> numpy array of accuracy values (float), aligned to ROC") get_accuracyROC;
%feature("autodoc", "get_errorROC(self) -> numpy array of error rate values (float), aligned to ROC") get_errorROC;
%feature("autodoc", "get_PRC(self) -> numpy array of Precision Recall Curve points (2 floats)") get_PRC;
%feature("autodoc", "get_fmeasurePRC(self) -> numpy array of F-measure values %(float), aligned to PRC") get_fmeasurePRC;
%feature("autodoc", "get_DET(self) -> numpy array of Detection Error Tradeoff points (2 floats)") get_DET;
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
 #include <shogun/evaluation/PerformanceMeasures.h>
%}

/* Typemaps */
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** result, int32_t* num)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** result, int32_t* num, int32_t* dim)};
/* workaround swig bug */
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(double_t** result, int32_t* num)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(double_t** result, int32_t* num, int32_t* dim)};

/* Remove C Prefix */
%rename(PerformanceMeasures) CPerformanceMeasures;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/evaluation/PerformanceMeasures.h>
