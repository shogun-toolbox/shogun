%{
 #include <multiclass/tree/TreeMachine.h>
 #include <multiclass/tree/RelaxedTreeNodeData.h>
 #include <multiclass/tree/ConditionalProbabilityTreeNodeData.h>
 #include <multiclass/tree/ConditionalProbabilityTree.h>
 #include <multiclass/tree/BalancedConditionalProbabilityTree.h>
 #include <multiclass/tree/RandomConditionalProbabilityTree.h>
 #include <multiclass/tree/RelaxedTree.h>
 #include <multiclass/tree/RelaxedTreeUtil.h>
 #include <multiclass/tree/TreeMachineNode.h>
 #include <multiclass/tree/VwConditionalProbabilityTree.h>

 #include <multiclass/RejectionStrategy.h>
 #include <multiclass/MulticlassStrategy.h>
 #include <multiclass/MulticlassOneVsRestStrategy.h>
 #include <multiclass/MulticlassOneVsOneStrategy.h>
 #include <machine/BaseMulticlassMachine.h>
 #include <machine/MulticlassMachine.h>
 #include <machine/NativeMulticlassMachine.h>
 #include <machine/LinearMulticlassMachine.h>
 #include <machine/KernelMulticlassMachine.h>
 #include <multiclass/MulticlassSVM.h>
 #include <classifier/mkl/MKLMulticlass.h>

 #include <multiclass/ecoc/ECOCStrategy.h>
 #include <multiclass/ecoc/ECOCEncoder.h>
 #include <multiclass/ecoc/ECOCOVOEncoder.h>
 #include <multiclass/ecoc/ECOCRandomSparseEncoder.h>
 #include <multiclass/ecoc/ECOCRandomDenseEncoder.h>
 #include <multiclass/ecoc/ECOCDiscriminantEncoder.h>
 #include <multiclass/ecoc/ECOCForestEncoder.h>
 #include <multiclass/ecoc/ECOCDecoder.h>
 #include <multiclass/ecoc/ECOCOVREncoder.h>
 #include <multiclass/ecoc/ECOCSimpleDecoder.h>
 #include <multiclass/ecoc/ECOCHDDecoder.h>
 #include <multiclass/ecoc/ECOCIHDDecoder.h>
 #include <multiclass/ecoc/ECOCEDDecoder.h>
 #include <multiclass/ecoc/ECOCAEDDecoder.h>
 #include <multiclass/ecoc/ECOCLLBDecoder.h>

 #include <multiclass/MulticlassTreeGuidedLogisticRegression.h>
 #include <multiclass/MulticlassLogisticRegression.h>
 #include <multiclass/MulticlassLibLinear.h>
 #include <multiclass/MulticlassOCAS.h>
 #include <multiclass/MulticlassSVM.h>
 #include <multiclass/LaRank.h>
 #include <multiclass/MulticlassLibSVM.h>
 #include <multiclass/GMNPSVM.h>
 #include <multiclass/ScatterSVM.h>
 #include <multiclass/KNN.h>
 #include <multiclass/GaussianNaiveBayes.h>
 #include <multiclass/QDA.h>
 #include <multiclass/MCLDA.h>
 #include <multiclass/ShareBoost.h>
%}
