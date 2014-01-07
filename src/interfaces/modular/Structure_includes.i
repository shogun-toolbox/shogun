%{
 #include <structure/PlifBase.h>
 #include <structure/Plif.h>
 #include <structure/PlifArray.h>
 #include <structure/DynProg.h>
 #include <structure/PlifMatrix.h>
 #include <structure/IntronList.h>
 #include <structure/SegmentLoss.h>

 #include <structure/BmrmStatistics.h>
 #include <structure/StructuredModel.h>
 #include <structure/MulticlassModel.h>
 #include <structure/MulticlassSOLabels.h>
 #include <structure/HMSVMModel.h>
 #include <structure/SequenceLabels.h>
 #include <structure/StateModelTypes.h>
 #include <structure/StateModel.h>
 #include <structure/TwoStateModel.h>
 #include <structure/DirectorStructuredModel.h>

 #include <structure/FactorType.h>
 #include <structure/Factor.h>
 #include <structure/DisjointSet.h>
 #include <structure/FactorGraph.h>
 #include <features/FactorGraphFeatures.h>
 #include <labels/FactorGraphLabels.h>
 #include <structure/MAPInference.h>
 #include <structure/FactorGraphModel.h>

 #include <structure/SOSVMHelper.h>
 #include <machine/StructuredOutputMachine.h>
 #include <machine/LinearStructuredOutputMachine.h>
 #include <machine/KernelStructuredOutputMachine.h>

 #include <structure/DualLibQPBMSOSVM.h>

#ifdef USE_MOSEK
 #include <structure/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */

 #include <structure/StochasticSOSVM.h>
%}

