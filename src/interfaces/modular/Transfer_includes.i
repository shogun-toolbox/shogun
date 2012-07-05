%{
 #include <shogun/transfer/multitask/MultitaskKernelNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelMklNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelTreeNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelMaskNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelMaskPairNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelPlifNormalizer.h>

 #include <shogun/transfer/multitask/LibLinearMTL.h>
 #include <shogun/transfer/multitask/Task.h>
 #include <shogun/transfer/multitask/TaskTree.h>
 #include <shogun/transfer/multitask/MultitaskLSRegression.h>
 #include <shogun/transfer/multitask/MultitaskLogisticRegression.h>

#ifdef USE_SVMLIGHT
 #include <shogun/transfer/domain_adaptation/DomainAdaptationSVM.h>
#endif /* USE_SVMLIGHT */
 #include <shogun/transfer/domain_adaptation/DomainAdaptationSVMLinear.h>
%}
