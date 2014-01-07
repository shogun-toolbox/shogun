%{
 #include <transfer/multitask/MultitaskKernelNormalizer.h>
 #include <transfer/multitask/MultitaskKernelMklNormalizer.h>
 #include <transfer/multitask/MultitaskKernelTreeNormalizer.h>
 #include <transfer/multitask/MultitaskKernelMaskNormalizer.h>
 #include <transfer/multitask/MultitaskKernelMaskPairNormalizer.h>
 #include <transfer/multitask/MultitaskKernelPlifNormalizer.h>

 #include <transfer/multitask/LibLinearMTL.h>
 #include <transfer/multitask/Task.h>
 #include <transfer/multitask/TaskRelation.h>
 #include <transfer/multitask/TaskTree.h>
 #include <transfer/multitask/TaskGroup.h>
 #include <transfer/multitask/MultitaskLinearMachine.h>
 #include <transfer/multitask/MultitaskLeastSquaresRegression.h>
 #include <transfer/multitask/MultitaskLogisticRegression.h>
 #include <transfer/multitask/MultitaskL12LogisticRegression.h>
 #include <transfer/multitask/MultitaskTraceLogisticRegression.h>
 #include <transfer/multitask/MultitaskClusteredLogisticRegression.h>

 #include <transfer/multitask/MultitaskROCEvaluation.h>

#ifdef USE_SVMLIGHT
 #include <transfer/domain_adaptation/DomainAdaptationSVM.h>
#endif /* USE_SVMLIGHT */
 #include <transfer/domain_adaptation/DomainAdaptationSVMLinear.h>
 #include <transfer/domain_adaptation/DomainAdaptationMulticlassLibLinear.h>
%}
