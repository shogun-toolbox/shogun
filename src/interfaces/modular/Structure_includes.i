%{
 #include <shogun/structure/PlifBase.h>
 #include <shogun/structure/Plif.h>
 #include <shogun/structure/PlifArray.h>
 #include <shogun/structure/DynProg.h>
 #include <shogun/structure/PlifMatrix.h>
 #include <shogun/structure/IntronList.h>
 #include <shogun/structure/SegmentLoss.h>

 #include <shogun/structure/BmrmStatistics.h>
 #include <shogun/structure/StructuredModel.h>
 #include <shogun/structure/MulticlassModel.h>
 #include <shogun/structure/MulticlassSOLabels.h>
 #include <shogun/structure/HMSVMModel.h>
 #include <shogun/structure/SequenceLabels.h>
 #include <shogun/structure/StateModelTypes.h>
 #include <shogun/structure/StateModel.h>
 #include <shogun/structure/TwoStateModel.h>
 #include <shogun/structure/DirectorStructuredModel.h>
 #include <shogun/structure/MultilabelSOLabels.h>
 #include <shogun/structure/MultilabelModel.h>
 #include <shogun/structure/HashedMultilabelModel.h>
 #include <shogun/structure/MultilabelCLRModel.h>

 #include <shogun/structure/FactorType.h>
 #include <shogun/structure/Factor.h>
 #include <shogun/structure/DisjointSet.h>
 #include <shogun/structure/FactorGraph.h>
 #include <shogun/features/FactorGraphFeatures.h>
 #include <shogun/labels/FactorGraphLabels.h>
 #include <shogun/structure/MAPInference.h>
 #include <shogun/structure/GraphCut.h>
 #include <shogun/structure/FactorGraphModel.h>

 #include <shogun/structure/SOSVMHelper.h>
 #include <shogun/machine/StructuredOutputMachine.h>
 #include <shogun/machine/LinearStructuredOutputMachine.h>
 #include <shogun/machine/KernelStructuredOutputMachine.h>

 #include <shogun/structure/DualLibQPBMSOSVM.h>

#ifdef USE_MOSEK
 #include <shogun/structure/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */

 #include <shogun/structure/StochasticSOSVM.h>
 #include <shogun/structure/FWSOSVM.h>
%}

