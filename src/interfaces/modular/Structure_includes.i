%{
 #include <shogun/structure/RiskFunction.h>
 #include <shogun/structure/DirectorRiskFunction.h>
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
 #include <shogun/structure/HMSVMModel.h>
 #include <shogun/structure/HMSVMLabels.h>
 #include <shogun/structure/StateModelTypes.h>
 #include <shogun/structure/StateModel.h>
 #include <shogun/structure/TwoStateModel.h>

 #include <shogun/structure/libbmrm.h>
 #include <shogun/structure/libppbm.h>
 #include <shogun/structure/libp3bm.h>
 #include <shogun/structure/MulticlassRiskFunction.h>
 #include <shogun/structure/DualLibQPBMSOSVM.h>

#ifdef USE_MOSEK
 #include <shogun/structure/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */
%}

