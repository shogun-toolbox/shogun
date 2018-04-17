/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Saloni Nigam, Sergey Lisitsyn
 */

/* Remove C Prefix */
%rename(BalancedConditionalProbabilityTree) CBalancedConditionalProbabilityTree;
%rename(ConditionalProbabilityTree) CConditionalProbabilityTree;
%rename(RandomConditionalProbabilityTree) CRandomConditionalProbabilityTree;
%rename(RelaxedTree) CRelaxedTree;
%rename(TreeMachineNode) CTreeMachineNode;
%rename(ID3ClassifierTree) CID3ClassifierTree;
%rename(C45ClassifierTree) CC45ClassifierTree;
%rename(CARTree) CCARTree;
%rename(CHAIDTree) CCHAIDTree;

%rename(RejectionStrategy) CRejectionStrategy;
%rename(ThresholdRejectionStrategy) CThresholdRejectionStrategy;
%rename(DixonQTestRejectionStrategy) CDixonQTestRejectionStrategy;
%rename(MulticlassStrategy) CMulticlassStrategy;
%rename(MulticlassOneVsRestStrategy) CMulticlassOneVsRestStrategy;
%rename(MulticlassOneVsOneStrategy) CMulticlassOneVsOneStrategy;
%rename(BaseMulticlassMachine) CBaseMulticlassMachine;
%rename(MulticlassMachine) CMulticlassMachine;
%rename(NativeMulticlassMachine) CNativeMulticlassMachine;
%rename(LinearMulticlassMachine) CLinearMulticlassMachine;
%rename(KernelMulticlassMachine) CKernelMulticlassMachine;
%rename(MulticlassSVM) CMulticlassSVM;
%rename(MKLMulticlass) CMKLMulticlass;

%newobject apply_multilabel_output();

%rename(ECOCStrategy) CECOCStrategy;
%rename(ECOCEncoder) CECOCEncoder;
%rename(ECOCDecoder) CECOCDecoder;

#ifdef USE_GPL_SHOGUN
%rename(MulticlassTreeGuidedLogisticRegression) CMulticlassTreeGuidedLogisticRegression;
%rename(MulticlassLogisticRegression) CMulticlassLogisticRegression;
%rename(MulticlassOCAS) CMulticlassOCAS;
#endif //USE_GPL_SHOGUN
%rename(MulticlassLibLinear) CMulticlassLibLinear;
%rename(MulticlassSVM) CMulticlassSVM;
%rename(MulticlassLibSVM) CMulticlassLibSVM;

%rename(LaRank) CLaRank;
%rename(ScatterSVM) CScatterSVM;
%rename(GMNPSVM) CGMNPSVM;
%rename(KNN) CKNN;
%rename(GaussianNaiveBayes) CGaussianNaiveBayes;
%rename(QDA) CQDA;
%rename(MCLDA) CMCLDA;

%rename(ShareBoost) CShareBoost;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/machine/BaseMulticlassMachine.h>
%include <shogun/multiclass/tree/TreeMachine.h>
%include <shogun/multiclass/tree/RelaxedTreeNodeData.h>
%include <shogun/multiclass/tree/ConditionalProbabilityTreeNodeData.h>
namespace shogun
{
    %template(TreeMachineWithConditionalProbabilityTreeNodeData) CTreeMachine<ConditionalProbabilityTreeNodeData>;
    %template(TreeMachineWithRelaxedTreeNodeData) CTreeMachine<RelaxedTreeNodeData>;
    %template(TreeMachineWithID3TreeNodeData) CTreeMachine<id3TreeNodeData>;
    %template(TreeMachineWithC45TreeNodeData) CTreeMachine<C45TreeNodeData>;
    %template(TreeMachineWithCARTreeNodeData) CTreeMachine<CARTreeNodeData>;
    %template(TreeMachineWithCHAIDTreeNodeData) CTreeMachine<CHAIDTreeNodeData>;
}

%include <shogun/multiclass/tree/ConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/BalancedConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/RandomConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/RelaxedTree.h>
%include <shogun/multiclass/tree/TreeMachineNode.h>

%include <shogun/multiclass/tree/ID3ClassifierTree.h>
%include <shogun/multiclass/tree/C45ClassifierTree.h>
%include <shogun/multiclass/tree/CARTree.h>
%include <shogun/multiclass/tree/CHAIDTree.h>

%include <shogun/multiclass/RejectionStrategy.h>
%include <shogun/multiclass/MulticlassStrategy.h>
%include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
%include <shogun/multiclass/MulticlassOneVsOneStrategy.h>
%include <shogun/machine/MulticlassMachine.h>
%include <shogun/machine/NativeMulticlassMachine.h>
%include <shogun/machine/LinearMulticlassMachine.h>
%include <shogun/machine/KernelMulticlassMachine.h>
%include <shogun/multiclass/MulticlassSVM.h>
%include <shogun/classifier/mkl/MKLMulticlass.h>

%include <shogun/multiclass/ecoc/ECOCEncoder.h>
%include <shogun/multiclass/ecoc/ECOCDecoder.h>
%include <shogun/multiclass/ecoc/ECOCStrategy.h>

#ifdef USE_GPL_SHOGUN
%include <shogun/multiclass/MulticlassTreeGuidedLogisticRegression.h>
%include <shogun/multiclass/MulticlassLogisticRegression.h>
%include <shogun/multiclass/MulticlassOCAS.h>
%include <shogun/multiclass/LaRank.h>
#endif // USE_GPL_SHOGUN
%include <shogun/multiclass/MulticlassLibLinear.h>
%include <shogun/multiclass/MulticlassSVM.h>
%include <shogun/multiclass/MulticlassLibSVM.h>
%include <shogun/multiclass/ScatterSVM.h>
%include <shogun/multiclass/GMNPSVM.h>
%include <shogun/multiclass/KNN.h>
%include <shogun/multiclass/GaussianNaiveBayes.h>
%include <shogun/multiclass/QDA.h>
%include <shogun/multiclass/MCLDA.h>
%include <shogun/multiclass/ShareBoost.h>
