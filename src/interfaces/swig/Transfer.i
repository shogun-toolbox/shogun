/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn
 */

/* Multitask renames */
%shared_ptr(shogun::Node)
%shared_ptr(shogun::MultitaskLinearMachine)
%shared_ptr(shogun::MultitaskKernelNormalizer)
%shared_ptr(shogun::MultitaskKernelMklNormalizer)
%shared_ptr(shogun::MultitaskKernelTreeNormalizer)
%shared_ptr(shogun::MultitaskKernelMaskNormalizer)
%shared_ptr(shogun::MultitaskKernelMaskPairNormalizer)
%shared_ptr(shogun::MultitaskKernelPlifNormalizer)

%shared_ptr(shogun::Task)
%shared_ptr(shogun::TaskRelation)
%shared_ptr(shogun::TaskRelationBase)
%shared_ptr(shogun::TaskTree)
%shared_ptr(shogun::TaskGroup)
#ifdef USE_GPL_SHOGUN
%shared_ptr(shogun::MultitaskLinearMachineBase)
%shared_ptr(shogun::MultitaskLeastSquaresRegression)
%shared_ptr(shogun::MultitaskLogisticRegression)
%shared_ptr(shogun::MultitaskL12LogisticRegression)
%shared_ptr(shogun::MultitaskTraceLogisticRegression)
%shared_ptr(shogun::MultitaskClusteredLogisticRegression)
#endif //USE_GPL_SHOGUN

%shared_ptr(shogun::MultitaskROCEvaluation)

%shared_ptr(shogun::LibLinearMTL)

/* Domain adaptation renames */
%shared_ptr(shogun::DomainAdaptationSVMLinear)
%shared_ptr(shogun::DomainAdaptationMulticlassLibLinear)

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
#ifdef USE_GPL_SHOGUN
%include <shogun/transfer/multitask/MultitaskLinearMachine.h>
%include <shogun/transfer/multitask/MultitaskLeastSquaresRegression.h>
%include <shogun/transfer/multitask/MultitaskLogisticRegression.h>
%include <shogun/transfer/multitask/MultitaskL12LogisticRegression.h>
%include <shogun/transfer/multitask/MultitaskTraceLogisticRegression.h>
#endif //USE_GPL_SHOGUN

%include <shogun/transfer/multitask/MultitaskROCEvaluation.h>
%include <shogun/transfer/multitask/LibLinearMTL.h>

#ifdef USE_GPL_SHOGUN
%include <shogun/transfer/multitask/MultitaskClusteredLogisticRegression.h>
#endif // USE_GPL_SHOGUN

/* Domain adaptation includes */
%include <shogun/transfer/domain_adaptation/DomainAdaptationSVMLinear.h>
%include <shogun/transfer/domain_adaptation/DomainAdaptationMulticlassLibLinear.h>
