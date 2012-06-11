%{
 #include <shogun/machine/StructuredOutputMachine.h>
 #include <shogun/machine/LinearStructuredOutputMachine.h>

 #include <shogun/so/StructuredModel.h>
 #include <shogun/so/MulticlassModel.h>
 #include <shogun/so/MulticlassSOLabels.h>

#ifdef USE_MOSEK
 #include <shogun/so/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */
%}
