/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
 
/* Remove C Prefix */
%rename(Evaluation) CEvaluation;
%rename(BinaryClassEvaluation) CBinaryClassEvaluation;
%rename(ClusteringEvaluation) CClusteringEvaluation;
%rename(ClusteringAccuracy) CClusteringAccuracy;
%rename(ContingencyTableEvaluation) CContingencyTableEvaluation;
%rename(MulticlassAccuracy) CMulticlassAccuracy;
%rename(MeanAbsoluteError) CMeanAbsoluteError;
%rename(MeanSquaredError) CMeanSquaredError;
%rename(MeanSquaredLogError) CMeanSquaredLogError;
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
%rename(CrossValidation) CCrossValidation;
%rename(SplittingStrategy) CSplittingStrategy;
%rename(StratifiedCrossValidationSplitting) CStratifiedCrossValidationSplitting;
%rename(CrossValidationSplitting) CCrossValidationSplitting;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/evaluation/Evaluation.h>
%include <shogun/evaluation/BinaryClassEvaluation.h>
%include <shogun/evaluation/ClusteringEvaluation.h>
%include <shogun/evaluation/ClusteringAccuracy.h>
%include <shogun/evaluation/ContingencyTableEvaluation.h>
%include <shogun/evaluation/MulticlassAccuracy.h>
%include <shogun/evaluation/MeanAbsoluteError.h>
%include <shogun/evaluation/MeanSquaredError.h>
%include <shogun/evaluation/MeanSquaredLogError.h>
%include <shogun/evaluation/ROCEvaluation.h>
%include <shogun/evaluation/PRCEvaluation.h>
%include <shogun/evaluation/CrossValidation.h>
%include <shogun/evaluation/SplittingStrategy.h>
%include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
%include <shogun/evaluation/CrossValidationSplitting.h>
