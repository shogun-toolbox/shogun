%{
 #include <shogun/transfer/multitask/MultitaskKernelNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelMklNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelTreeNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelMaskNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelMaskPairNormalizer.h>
 #include <shogun/transfer/multitask/MultitaskKernelPlifNormalizer.h>

#ifdef USE_SVMLIGHT
 #include <shogun/transfer/domain_adaptation/DomainAdaptationSVM.h>
#endif /* USE_SVMLIGHT */
 #include <shogun/transfer/domain_adaptation/DomainAdaptationSVMLinear.h>
%}
