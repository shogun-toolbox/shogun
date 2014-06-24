/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

/* These functions return new Objects */
%newobject CGradientEvaluation::evaluate();
%newobject CCrossValidation::evaluate();

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::CDirectorContingencyTableEvaluation;
#endif

/* Remove C Prefix */
%rename(Evaluation) CEvaluation;
%rename(BinaryClassEvaluation) CBinaryClassEvaluation;
%rename(ClusteringEvaluation) CClusteringEvaluation;
%rename(ClusteringAccuracy) CClusteringAccuracy;
%rename(ClusteringMutualInformation) CClusteringMutualInformation;
%rename(ContingencyTableEvaluation) CContingencyTableEvaluation;
%rename(MulticlassAccuracy) CMulticlassAccuracy;
%rename(MultilabelAccuracy) CMultilabelAccuracy;
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
%rename(DifferentiableFunction) CDifferentiableFunction;
%rename(GradientCriterion) CGradientCriterion;
%rename(GradientEvaluation) CGradientEvaluation;
%rename(MulticlassOVREvaluation) CMulticlassOVREvaluation;
%rename(CrossValidationResult) CCrossValidationResult;
%rename(CrossValidationOutput) CCrossValidationOutput;
%rename(CrossValidationPrintOutput) CCrossValidationPrintOutput;
%rename(CrossValidationMKLStorage) CCrossValidationMKLStorage;
%rename(CrossValidationMulticlassStorage) CCrossValidationMulticlassStorage;
%rename(StructuredAccuracy) CStructuredAccuracy;
%rename(DirectorContingencyTableEvaluation) CDirectorContingencyTableEvaluation;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/evaluation/EvaluationResult.h>
%include <shogun/evaluation/Evaluation.h>
%include <shogun/evaluation/BinaryClassEvaluation.h>
%include <shogun/evaluation/ClusteringEvaluation.h>
%include <shogun/evaluation/ClusteringAccuracy.h>
%include <shogun/evaluation/ClusteringMutualInformation.h>
%include <shogun/evaluation/ContingencyTableEvaluation.h>
%include <shogun/evaluation/MulticlassAccuracy.h>
%include <shogun/evaluation/MultilabelAccuracy.h>
%include <shogun/evaluation/MeanAbsoluteError.h>
%include <shogun/evaluation/MeanSquaredError.h>
%include <shogun/evaluation/MeanSquaredLogError.h>
%include <shogun/evaluation/ROCEvaluation.h>
%include <shogun/evaluation/PRCEvaluation.h>
%include <shogun/evaluation/MachineEvaluation.h>
%include <shogun/evaluation/CrossValidation.h>
%include <shogun/evaluation/SplittingStrategy.h>
%include <shogun/evaluation/DifferentiableFunction.h>
%include <shogun/evaluation/GradientCriterion.h>
%include <shogun/evaluation/GradientEvaluation.h>
%include <shogun/evaluation/GradientResult.h>
%include <shogun/evaluation/MulticlassOVREvaluation.h>
%include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
%include <shogun/evaluation/CrossValidationSplitting.h>
%include <shogun/evaluation/CrossValidationOutput.h>
%include <shogun/evaluation/CrossValidationPrintOutput.h>
%include <shogun/evaluation/CrossValidationMKLStorage.h>
%include <shogun/evaluation/CrossValidationMulticlassStorage.h>
%include <shogun/evaluation/StructuredAccuracy.h>
%include <shogun/evaluation/DirectorContingencyTableEvaluation.h>
