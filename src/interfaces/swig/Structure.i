/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn
 */

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::DirectorStructuredModel;
#endif

/* Remove C Prefix */
%shared_ptr(shogun::PlifBase)
%shared_ptr(shogun::Plif)
%shared_ptr(shogun::PlifArray)
%shared_ptr(shogun::DynProg)
%shared_ptr(shogun::PlifMatrix)
%shared_ptr(shogun::SegmentLoss)
%shared_ptr(shogun::IntronList)

%shared_ptr(shogun::StructuredModel)
%shared_ptr(shogun::ResultSet)
%shared_ptr(shogun::MulticlassModel)
%shared_ptr(shogun::MulticlassSOLabels)
%shared_ptr(shogun::RealNumber)
%shared_ptr(shogun::HMSVMModel)
%shared_ptr(shogun::SequenceLabels)
%shared_ptr(shogun::Sequence)
%shared_ptr(shogun::StateModel)
%shared_ptr(shogun::TwoStateModel)
%shared_ptr(shogun::DirectorStructuredModel)
%shared_ptr(shogun::MultilabelSOLabels)
%shared_ptr(shogun::SparseMultilabel)
%shared_ptr(shogun::MultilabelModel)
%shared_ptr(shogun::HashedMultilabelModel)
%shared_ptr(shogun::MultilabelCLRModel)
%shared_ptr(shogun::HierarchicalMultilabelModel)

%shared_ptr(shogun::FactorType)
%shared_ptr(shogun::TableFactorType)
%shared_ptr(shogun::FactorDataSource)
%shared_ptr(shogun::Factor)
%shared_ptr(shogun::DisjointSet)
%shared_ptr(shogun::FactorGraph)
%shared_ptr(shogun::FactorGraphObservation)
%shared_ptr(shogun::FactorGraphLabels)
%shared_ptr(shogun::FactorGraphFeatures)
%shared_ptr(shogun::MAPInference)
%shared_ptr(shogun::MAPInferImpl)
%shared_ptr(shogun::GraphCut)
%shared_ptr(shogun::FactorGraphModel)

%shared_ptr(shogun::SOSVMHelper)
%shared_ptr(shogun::StructuredOutputMachine)
%shared_ptr(shogun::LinearStructuredOutputMachine)
%shared_ptr(shogun::KernelStructuredOutputMachine)

#ifdef USE_GPL_SHOGUN
%shared_ptr(shogun::DualLibQPBMSOSVM)
#ifdef USE_MOSEK
%shared_ptr(shogun::PrimalMosekSOSVM)
#endif /* USE_MOSEK */
#endif //USE_GPL_SHOGUN

%shared_ptr(shogun::StochasticSOSVM)
%shared_ptr(shogun::FWSOSVM)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/structure/PlifBase.h>
%include <shogun/structure/Plif.h>
%include <shogun/structure/PlifArray.h>
%include <shogun/structure/DynProg.h>
%include <shogun/structure/PlifMatrix.h>
%include <shogun/structure/IntronList.h>
%include <shogun/structure/SegmentLoss.h>

#ifdef USE_GPL_SHOGUN
%include <shogun/structure/BmrmStatistics.h>
#endif //USE_GPL_SHOGUN
%include <shogun/structure/StructuredModel.h>
%include <shogun/structure/MulticlassModel.h>
%include <shogun/structure/MulticlassSOLabels.h>
%include <shogun/structure/HMSVMModel.h>
%include <shogun/structure/SequenceLabels.h>
%include <shogun/structure/StateModelTypes.h>
%include <shogun/structure/StateModel.h>
%include <shogun/structure/TwoStateModel.h>
%include <shogun/structure/DirectorStructuredModel.h>
%include <shogun/structure/MultilabelSOLabels.h>
%include <shogun/structure/MultilabelModel.h>
%include <shogun/structure/HashedMultilabelModel.h>
%include <shogun/structure/MultilabelCLRModel.h>
%include <shogun/structure/HierarchicalMultilabelModel.h>

%include <shogun/structure/FactorType.h>
%include <shogun/structure/Factor.h>
%include <shogun/structure/DisjointSet.h>
%include <shogun/structure/FactorGraph.h>
%include <shogun/features/FactorGraphFeatures.h>
%include <shogun/labels/FactorGraphLabels.h>
%include <shogun/structure/MAPInference.h>
%include <shogun/structure/GraphCut.h>
%include <shogun/structure/FactorGraphModel.h>

%include <shogun/structure/SOSVMHelper.h>
%include <shogun/machine/StructuredOutputMachine.h>
%include <shogun/machine/LinearStructuredOutputMachine.h>

/** Instantiate RandomMixin */
%template(SeedableLinearStructured) shogun::Seedable<shogun::LinearStructuredOutputMachine>;
%template(RandomMixinLinearStructured) shogun::RandomMixin<shogun::LinearStructuredOutputMachine, std::mt19937_64>;

%include <shogun/machine/KernelStructuredOutputMachine.h>

#ifdef USE_GPL_SHOGUN
%include <shogun/structure/DualLibQPBMSOSVM.h>

#ifdef USE_MOSEK
%include <shogun/structure/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */
#endif //USE_GPL_SHOGUN

%include <shogun/structure/StochasticSOSVM.h>
%include <shogun/structure/FWSOSVM.h>
