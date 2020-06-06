/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Saloni Nigam, Sergey Lisitsyn
 */

%shared_ptr(shogun::BaseMulticlassMachine)

%shared_ptr(shogun::TreeMachineNode)
%shared_ptr(shogun::TreeMachine<shogun::ConditionalProbabilityTreeNodeData>)
%shared_ptr(shogun::TreeMachine<shogun::RelaxedTreeNodeData>)
%shared_ptr(shogun::TreeMachine<shogun::id3TreeNodeData>)
%shared_ptr(shogun::TreeMachine<shogun::C45TreeNodeData>)
%shared_ptr(shogun::Seedable<shogun::TreeMachine<shogun::CARTreeNodeData>>)
%shared_ptr(shogun::RandomMixin<shogun::TreeMachine<shogun::CARTreeNodeData>, std::mt19937_64>)
%shared_ptr(shogun::TreeMachine<shogun::CARTreeNodeData>)
%shared_ptr(shogun::TreeMachine<shogun::CHAIDTreeNodeData>)
%shared_ptr(shogun::FeatureImportanceTree<shogun::C45TreeNodeData>)
%shared_ptr(shogun::FeatureImportanceTree<shogun::id3TreeNodeData>)
%shared_ptr(shogun::RandomMixin<shogun::FeatureImportanceTree<shogun::CARTreeNodeData>, std::mt19937_64>)

%shared_ptr(shogun::BalancedConditionalProbabilityTree)
%shared_ptr(shogun::ConditionalProbabilityTree)
SHARED_RANDOM_INTERFACE(shogun::ConditionalProbabilityTree)
%shared_ptr(shogun::RandomConditionalProbabilityTree)
%shared_ptr(shogun::RelaxedTree)

%shared_ptr(shogun::ID3ClassifierTree)
%shared_ptr(shogun::C45ClassifierTree)
%shared_ptr(shogun::CARTree)
%shared_ptr(shogun::CHAIDTree)

%shared_ptr(shogun::RejectionStrategy)
%shared_ptr(shogun::ThresholdRejectionStrategy)
%shared_ptr(shogun::DixonQTestRejectionStrategy)
%shared_ptr(shogun::MulticlassStrategy)
%shared_ptr(shogun::LinearMulticlassMachine)


%shared_ptr(shogun::ECOCStrategy)
%shared_ptr(shogun::ECOCEncoder)
%shared_ptr(shogun::ECOCDecoder)
%shared_ptr(shogun::ECOCOVREncoder)
%shared_ptr(shogun::ECOCOVOEncoder)
SHARED_RANDOM_INTERFACE(shogun::ECOCEncoder)
%shared_ptr(shogun::ECOCRandomSparseEncoder)
%shared_ptr(shogun::ECOCRandomDenseEncoder)
%shared_ptr(shogun::ECOCDiscriminantEncoder)
%shared_ptr(shogun::ECOCForestEncoder)
%shared_ptr(shogun::ECOCSimpleDecoder)
%shared_ptr(shogun::ECOCHDDecoder)
%shared_ptr(shogun::ECOCIHDDecoder)
%shared_ptr(shogun::ECOCEDDecoder)
%shared_ptr(shogun::ECOCAEDDecoder)
%shared_ptr(shogun::ECOCLLBDecoder)

#ifdef USE_GPL_SHOGUN
%shared_ptr(shogun::MulticlassTreeGuidedLogisticRegression)
%shared_ptr(shogun::MulticlassLogisticRegression)
#endif //USE_GPL_SHOGUN
%shared_ptr(shogun::MulticlassLibLinear)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/machine/BaseMulticlassMachine.h>
%include <shogun/multiclass/tree/TreeMachine.h>
%include <shogun/multiclass/tree/RelaxedTreeNodeData.h>
%include <shogun/multiclass/tree/ConditionalProbabilityTreeNodeData.h>
%include <shogun/multiclass/tree/FeatureImportanceTree.h>
namespace shogun
{
    %template(TreeMachineWithConditionalProbabilityTreeNodeData) TreeMachine<ConditionalProbabilityTreeNodeData>;
    %template(TreeMachineWithRelaxedTreeNodeData) TreeMachine<RelaxedTreeNodeData>;
}

%include <shogun/multiclass/tree/ConditionalProbabilityTree.h>
RANDOM_INTERFACE(ConditionalProbabilityTree)
%include <shogun/multiclass/tree/BalancedConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/RandomConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/RelaxedTree.h>
%include <shogun/multiclass/tree/TreeMachineNode.h>

%include <shogun/multiclass/tree/ID3TreeNodeData.h>
%template(TreeMachineWithID3TreeNodeData) shogun::TreeMachine<shogun::id3TreeNodeData>;
%template(FeatureImportanceTreeWithID3TreeNodeData) shogun::FeatureImportanceTree<shogun::id3TreeNodeData>;
%include <shogun/multiclass/tree/ID3ClassifierTree.h>
%include <shogun/multiclass/tree/C45TreeNodeData.h>
%template(TreeMachineWithC45TreeNodeData) shogun::TreeMachine<shogun::C45TreeNodeData>;
%template(FeatureImportanceTreeWithC45TreeNodeData) shogun::FeatureImportanceTree<shogun::C45TreeNodeData>;
%include <shogun/multiclass/tree/C45ClassifierTree.h>
%include <shogun/multiclass/tree/CARTreeNodeData.h>
%template(TreeMachineWithCARTreeNodeData) shogun::TreeMachine<shogun::CARTreeNodeData>;
%template(SeedableTreeMachine) shogun::Seedable<shogun::TreeMachine<shogun::CARTreeNodeData>>;
%template(RandomMixinTreeMachine) shogun::RandomMixin<shogun::TreeMachine<shogun::CARTreeNodeData>, std::mt19937_64>;
%template(RandomMixinFeatureImportanceTree) shogun::RandomMixin<shogun::FeatureImportanceTree<shogun::CARTreeNodeData>, std::mt19937_64>;
%include <shogun/multiclass/tree/CARTree.h>
%include <shogun/multiclass/tree/CHAIDTreeNodeData.h>
%template(TreeMachineWithCHAIDTreeNodeData) shogun::TreeMachine<shogun::CHAIDTreeNodeData>;
%include <shogun/multiclass/tree/CHAIDTree.h>

%include <shogun/multiclass/RejectionStrategy.h>
%include <shogun/multiclass/MulticlassStrategy.h>

%include <shogun/multiclass/ecoc/ECOCEncoder.h>
RANDOM_INTERFACE(ECOCEncoder)

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

#ifdef USE_GPL_SHOGUN
%include <shogun/multiclass/MulticlassTreeGuidedLogisticRegression.h>
%include <shogun/multiclass/MulticlassLogisticRegression.h>
#endif // USE_GPL_SHOGUN
