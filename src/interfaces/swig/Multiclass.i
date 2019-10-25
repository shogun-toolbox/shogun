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

%shared_ptr(shogun::BalancedConditionalProbabilityTree)
%shared_ptr(shogun::ConditionalProbabilityTree)
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
%shared_ptr(shogun::MulticlassOneVsRestStrategy)
%shared_ptr(shogun::MulticlassOneVsOneStrategy)
%shared_ptr(shogun::MulticlassMachine)
%shared_ptr(shogun::NativeMulticlassMachine)
%shared_ptr(shogun::LinearMulticlassMachine)
%shared_ptr(shogun::KernelMulticlassMachine)
%shared_ptr(shogun::MulticlassSVM)
%shared_ptr(shogun::MKLMulticlass)

%shared_ptr(shogun::ECOCStrategy)
%shared_ptr(shogun::ECOCEncoder)
%shared_ptr(shogun::ECOCDecoder)
%shared_ptr(shogun::ECOCOVREncoder)
%shared_ptr(shogun::ECOCOVOEncoder)
%shared_ptr(shogun::Seedable<shogun::ECOCEncoder>)
%shared_ptr(shogun::RandomMixin<shogun::ECOCEncoder, std::mt19937_64>)
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
%shared_ptr(shogun::MulticlassOCAS)
#endif //USE_GPL_SHOGUN
%shared_ptr(shogun::MulticlassSVM)
%shared_ptr(shogun::MulticlassLibLinear)

%shared_ptr(shogun::LaRank)
%shared_ptr(shogun::ScatterSVM)
%shared_ptr(shogun::GMNPSVM)
%shared_ptr(shogun::KNN)
%shared_ptr(shogun::GaussianNaiveBayes)
%shared_ptr(shogun::QDA)
%shared_ptr(shogun::MCLDA)

%shared_ptr(shogun::ShareBoost)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/machine/BaseMulticlassMachine.h>
%include <shogun/multiclass/tree/TreeMachine.h>
%include <shogun/multiclass/tree/RelaxedTreeNodeData.h>
%include <shogun/multiclass/tree/ConditionalProbabilityTreeNodeData.h>
namespace shogun
{
    %template(TreeMachineWithConditionalProbabilityTreeNodeData) TreeMachine<ConditionalProbabilityTreeNodeData>;
    %template(TreeMachineWithRelaxedTreeNodeData) TreeMachine<RelaxedTreeNodeData>;
  /*  %template(TreeMachineWithID3TreeNodeData) TreeMachine<id3TreeNodeData>;
    %template(TreeMachineWithC45TreeNodeData) TreeMachine<C45TreeNodeData>;
    %template(TreeMachineWithCARTreeNodeData) TreeMachine<CARTreeNodeData>;
    %template(TreeMachineWithCHAIDTreeNodeData) TreeMachine<CHAIDTreeNodeData>;
*/
    /** Instantiate RandomMixin *
    %template(SeedableTreeMachine) Seedable<TreeMachine<CARTreeNodeData>>;
    %template(RandomMixinTreeMachine) RandomMixin<TreeMachine<CARTreeNodeData>, std::mt19937_64>;
*/
}

%include <shogun/multiclass/tree/ConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/BalancedConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/RandomConditionalProbabilityTree.h>
%include <shogun/multiclass/tree/RelaxedTree.h>
%include <shogun/multiclass/tree/TreeMachineNode.h>

%include <shogun/multiclass/tree/ID3ClassifierTree.h>
%template(TreeMachineWithID3TreeNodeData) shogun::TreeMachine<shogun::id3TreeNodeData>;
%include <shogun/multiclass/tree/C45ClassifierTree.h>
%template(TreeMachineWithC45TreeNodeData) shogun::TreeMachine<shogun::C45TreeNodeData>;
%include <shogun/multiclass/tree/CARTree.h>
%template(TreeMachineWithCARTreeNodeData) shogun::TreeMachine<shogun::CARTreeNodeData>;
%template(SeedableTreeMachine) shogun::Seedable<shogun::TreeMachine<shogun::CARTreeNodeData>>;
%template(RandomMixinTreeMachine) shogun::RandomMixin<shogun::TreeMachine<shogun::CARTreeNodeData>, std::mt19937_64>;
%include <shogun/multiclass/tree/CHAIDTree.h>
%template(TreeMachineWithCHAIDTreeNodeData) shogun::TreeMachine<shogun::CHAIDTreeNodeData>;

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
namespace shogun
{
    /** Instantiate RandomMixin */
    %template(SeedableECOCEncoder) Seedable<ECOCEncoder>;
    %template(RandomMixinECOCEncoder) RandomMixin<ECOCEncoder, std::mt19937_64>;
}

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
%include <shogun/multiclass/MulticlassOCAS.h>
%include <shogun/multiclass/LaRank.h>
#endif // USE_GPL_SHOGUN
%include <shogun/multiclass/MulticlassSVM.h>
%include <shogun/multiclass/ScatterSVM.h>
%include <shogun/multiclass/GMNPSVM.h>
%include <shogun/multiclass/KNN.h>
%include <shogun/multiclass/GaussianNaiveBayes.h>
%include <shogun/multiclass/QDA.h>
%include <shogun/multiclass/MCLDA.h>
%include <shogun/multiclass/ShareBoost.h>
