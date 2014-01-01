/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

/* Multitask renames */
%rename(MultitaskKernelNormalizer) CMultitaskKernelNormalizer;
%rename(MultitaskKernelMklNormalizer) CMultitaskKernelMklNormalizer;
%rename(MultitaskKernelTreeNormalizer) CMultitaskKernelTreeNormalizer;
%rename(MultitaskKernelMaskNormalizer) CMultitaskKernelMaskNormalizer;
%rename(MultitaskKernelMaskPairNormalizer) CMultitaskKernelMaskPairNormalizer;
%rename(MultitaskKernelPlifNormalizer) CMultitaskKernelPlifNormalizer;

%rename(Task) CTask;
%rename(TaskRelationBase) CTaskRelation;
%rename(TaskTree) CTaskTree;
%rename(TaskGroup) CTaskGroup;
%rename(MultitaskLinearMachineBase) CMultitaskLinearMachine;
%rename(MultitaskLeastSquaresRegression) CMultitaskLeastSquaresRegression;
%rename(MultitaskLogisticRegression) CMultitaskLogisticRegression;
%rename(MultitaskL12LogisticRegression) CMultitaskL12LogisticRegression;
%rename(MultitaskTraceLogisticRegression) CMultitaskTraceLogisticRegression;
%rename(MultitaskClusteredLogisticRegression) CMultitaskClusteredLogisticRegression;

%rename(MultitaskROCEvaluation) CMultitaskROCEvaluation;

%rename(LibLinearMTL) CLibLinearMTL;

/* Domain adaptation renames */
#ifdef USE_SVMLIGHT
%rename(DomainAdaptationSVM) CDomainAdaptationSVM;
#endif //USE_SVMLIGHT
%rename(DomainAdaptationSVMLinear) CDomainAdaptationSVMLinear;
%rename(DomainAdaptationMulticlassLibLinear) CDomainAdaptationMulticlassLibLinear;

/* Multitask includes */
%include <shogun/transfer/multitask/MultitaskKernelNormalizer.h>
%include <shogun/transfer/multitask/MultitaskKernelMklNormalizer.h>
%include <shogun/transfer/multitask/MultitaskKernelTreeNormalizer.h>
%include <shogun/transfer/multitask/MultitaskKernelMaskNormalizer.h>
%include <shogun/transfer/multitask/MultitaskKernelMaskPairNormalizer.h>
%include <shogun/transfer/multitask/MultitaskKernelPlifNormalizer.h>

%include <shogun/transfer/multitask/Task.h>
%include <shogun/transfer/multitask/TaskRelation.h>
%include <shogun/transfer/multitask/TaskTree.h>
%include <shogun/transfer/multitask/TaskGroup.h>
%include <shogun/transfer/multitask/MultitaskLinearMachine.h>
%include <shogun/transfer/multitask/MultitaskLeastSquaresRegression.h>
%include <shogun/transfer/multitask/MultitaskLogisticRegression.h>
%include <shogun/transfer/multitask/MultitaskL12LogisticRegression.h>
%include <shogun/transfer/multitask/MultitaskTraceLogisticRegression.h>
%include <shogun/transfer/multitask/MultitaskClusteredLogisticRegression.h>

%include <shogun/transfer/multitask/MultitaskROCEvaluation.h>
%include <shogun/transfer/multitask/LibLinearMTL.h>

/* Domain adaptation includes */
#ifdef USE_SVMLIGHT
%include <shogun/transfer/domain_adaptation/DomainAdaptationSVM.h>
#endif // USE_SVMLIGHT
%include <shogun/transfer/domain_adaptation/DomainAdaptationSVMLinear.h>
%include <shogun/transfer/domain_adaptation/DomainAdaptationMulticlassLibLinear.h>
