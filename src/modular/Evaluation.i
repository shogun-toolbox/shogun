/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
 
%define DOCSTR
"The `Evaluation` module is a collection of classes like PerformanceMeasures for the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Evaluation

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Evaluation_doxygen.i"
#endif
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
 #include <shogun/features/Labels.h>
 #include <shogun/evaluation/Evaluation.h>
 #include <shogun/evaluation/BinaryClassEvaluation.h>
 #include <shogun/evaluation/ContingencyTableEvaluation.h>
 #include <shogun/evaluation/MulticlassAccuracy.h>
 #include <shogun/evaluation/MeanSquaredError.h>
 #include <shogun/evaluation/ROCEvaluation.h>
 #include <shogun/evaluation/PRCEvaluation.h>
%}

/* Typemaps */
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** result, int32_t* num)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** result, int32_t* num, int32_t* dim)};
/* workaround swig bug */
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(double_t** result, int32_t* num)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(double_t** result, int32_t* num, int32_t* dim)};



/* Remove C Prefix */
%rename(BaseEvaluation) CEvaluation;
%rename(BinaryClassEvaluation) CBinaryClassEvaluation;
%rename(ContingencyTableEvaluation) CContingencyTableEvaluation;
%rename(MulticlassAccuracy) CMulticlassAccuracy;
%rename(MeanSquaredError) CMeanSquaredError;
%rename(ROCEvaluation) CROCEvaluation;
%rename(PRCEvaluation) CPRCEvaluation;
%rename(AccuracyMeasure) CAccuracyMeasure;
%rename(ErrorRateMeasure) CErrorRateMeasure;
%rename(BALMeasure) CBALMeasure;
%rename(WRACCMeasure) CWRACCMeasure;
%rename(F1Measure) CF1Measure;
%rename(CrossCorrelationMeasure) CCrossCorrelationMeasure;
%rename(RecallMeasure) CRecallMeasure;
%rename(PrecisionMeasure) CPrecisionMeasure;
%rename(SpecificityMeasure) CSpecificityMeasure;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/features/Labels.h>
%include <shogun/evaluation/Evaluation.h>
%include <shogun/evaluation/BinaryClassEvaluation.h>
%include <shogun/evaluation/ContingencyTableEvaluation.h>
%include <shogun/evaluation/MulticlassAccuracy.h>
%include <shogun/evaluation/MeanSquaredError.h>
%include <shogun/evaluation/ROCEvaluation.h>
%include <shogun/evaluation/PRCEvaluation.h>
