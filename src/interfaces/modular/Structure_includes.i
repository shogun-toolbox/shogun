%{
 #include <shogun/structure/PlifBase.h>
 #include <shogun/structure/Plif.h>
 #include <shogun/structure/PlifArray.h>
 #include <shogun/structure/DynProg.h>
 #include <shogun/structure/PlifMatrix.h>
 #include <shogun/structure/IntronList.h>
 #include <shogun/structure/SegmentLoss.h>

 #include <shogun/machine/StructuredOutputMachine.h>
 #include <shogun/machine/LinearStructuredOutputMachine.h>
 #include <shogun/machine/KernelStructuredOutputMachine.h>

 #include <shogun/structure/StructuredModel.h>
 #include <shogun/structure/MulticlassModel.h>
 #include <shogun/structure/MulticlassSOLabels.h>

#ifdef USE_MOSEK
 #include <shogun/structure/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */
%}

