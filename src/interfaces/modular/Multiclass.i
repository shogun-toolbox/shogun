/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

/* Remove C Prefix */
%rename(BalancedConditionalProbabilityTree) CBalancedConditionalProbabilityTree;
%rename(ConditionalProbabilityTree) CConditionalProbabilityTree;
%rename(RandomConditionalProbabilityTree) CRandomConditionalProbabilityTree;
%rename(RelaxedTree) CRelaxedTree;
%rename(RelaxedTreeNodeData) CRelaxedTreeNodeData;
%rename(TreeMachineNode) CTreeMachineNode;
%rename(VwConditionalProbabilityTree) VwConditionalProbabilityTree;
%rename(ID3ClassifierTree) CID3ClassifierTree;
%rename(C45ClassifierTree) CC45ClassifierTree;
%rename(CARTree) CCARTree;

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
%rename(ECOCOVREncoder) CECOCOVREncoder;
%rename(ECOCOVOEncoder) CECOCOVOEncoder;
%rename(ECOCRandomSparseEncoder) CECOCRandomSparseEncoder;
%rename(ECOCRandomDenseEncoder) CECOCRandomDenseEncoder;
%rename(ECOCDiscriminantEncoder) CECOCDiscriminantEncoder;
%rename(ECOCForestEncoder) CECOCForestEncoder;
%rename(ECOCSimpleDecoder) CECOCSimpleDecoder;
%rename(ECOCHDDecoder) CECOCHDDecoder;
%rename(ECOCIHDDecoder) CECOCIHDDecoder;
%rename(ECOCEDDecoder) CECOCEDDecoder;
%rename(ECOCAEDDecoder) CECOCAEDDecoder;
%rename(ECOCLLBDecoder) CECOCLLBDecoder;

%rename(MulticlassTreeGuidedLogisticRegression) CMulticlassTreeGuidedLogisticRegression;
%rename(MulticlassLogisticRegression) CMulticlassLogisticRegression;
%rename(MulticlassLibLinear) CMulticlassLibLinear;
%rename(MulticlassOCAS) CMulticlassOCAS;
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
}

%include <shogun/multiclass/tree/ConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/BalancedConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/RandomConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/RelaxedTree.h>
%include <shogun/multiclass/tree/TreeMachineNode.h>
%include <shogun/multiclass/tree/VwConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/ID3ClassifierTree.h>
%include <shogun/multiclass/tree/C45ClassifierTree.h>
%include <shogun/multiclass/tree/CARTree.h>

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
%include <shogun/multiclass/ecoc/ECOCOVREncoder.h>
%include <shogun/multiclass/ecoc/ECOCOVOEncoder.h>
%include <shogun/multiclass/ecoc/ECOCRandomSparseEncoder.h>
%include <shogun/multiclass/ecoc/ECOCRandomDenseEncoder.h>
%include <shogun/multiclass/ecoc/ECOCDiscriminantEncoder.h>
%include <shogun/multiclass/ecoc/ECOCForestEncoder.h>
%include <shogun/multiclass/ecoc/ECOCSimpleDecoder.h>
%include <shogun/multiclass/ecoc/ECOCHDDecoder.h>
%include <shogun/multiclass/ecoc/ECOCIHDDecoder.h>
%include <shogun/multiclass/ecoc/ECOCEDDecoder.h>
%include <shogun/multiclass/ecoc/ECOCAEDDecoder.h>
%include <shogun/multiclass/ecoc/ECOCLLBDecoder.h>
%include <shogun/multiclass/ecoc/ECOCStrategy.h>

%include <shogun/multiclass/MulticlassTreeGuidedLogisticRegression.h>
%include <shogun/multiclass/MulticlassLogisticRegression.h>
%include <shogun/multiclass/MulticlassLibLinear.h>
%include <shogun/multiclass/MulticlassOCAS.h>
%include <shogun/multiclass/MulticlassSVM.h>
%include <shogun/multiclass/MulticlassLibSVM.h>
%include <shogun/multiclass/LaRank.h>
%include <shogun/multiclass/ScatterSVM.h>
%include <shogun/multiclass/GMNPSVM.h>
%include <shogun/multiclass/KNN.h>
%include <shogun/multiclass/GaussianNaiveBayes.h>
%include <shogun/multiclass/QDA.h>
%include <shogun/multiclass/MCLDA.h>
%include <shogun/multiclass/ShareBoost.h>
