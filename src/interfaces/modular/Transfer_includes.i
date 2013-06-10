%{
 #include <shogun/transfer/multitask/MultitaskKernelNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelMklNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelTreeNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelMaskNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelMaskPairNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelPlifNormalizer.h>

 #include <shogun/transfer/multitask/LibLinearMTL.h>
 #include <shogun/transfer/multitask/Task.h>
 #include <shogun/transfer/multitask/TaskRelation.h>
 #include <shogun/transfer/multitask/TaskTree.h>
 #include <shogun/transfer/multitask/TaskGroup.h>
 #include <shogun/transfer/multitask/MultitaskLinearMachine.h>
 #include <shogun/transfer/multitask/MultitaskLeastSquaresRegression.h>
 #include <shogun/transfer/multitask/MultitaskLogisticRegression.h>
 #include <shogun/transfer/multitask/MultitaskL12LogisticRegression.h>
 #include <shogun/transfer/multitask/MultitaskTraceLogisticRegression.h>
 #include <shogun/transfer/multitask/MultitaskClusteredLogisticRegression.h>

 #include <shogun/transfer/multitask/MultitaskROCEvaluation.h>

#ifdef USE_SVMLIGHT
 #include <shogun/transfer/domain_adaptation/DomainAdaptationSVM.h>
#endif /* USE_SVMLIGHT */
 #include <shogun/transfer/domain_adaptation/DomainAdaptationSVMLinear.h>
 #include <shogun/transfer/domain_adaptation/DomainAdaptationMulticlassLibLinear.h>
%}
