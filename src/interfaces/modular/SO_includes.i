%{
 #include <shogun/so/StructuredModel.h>
 #include <shogun/so/StructuredLossFunction.h>
 #include <shogun/so/ArgMaxFunction.h>

#ifdef USE_MOSEK
 #include <shogun/so/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */
%}
