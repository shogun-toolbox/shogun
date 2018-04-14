/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn
 */

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::CDirectorStructuredModel;
#endif

/* Remove C Prefix */
%rename(PlifBase) CPlifBase;
%rename(Plif) CPlif;
%rename(PlifArray) CPlifArray;
%rename(DynProg) CDynProg;
%rename(PlifMatrix) CPlifMatrix;
%rename(SegmentLoss) CSegmentLoss;
%rename(IntronList) CIntronList;

%rename(StructuredModel) CStructuredModel;
%rename(ResultSet) CResultSet;
%rename(MulticlassModel) CMulticlassModel;
%rename(MulticlassSOLabels) CMulticlassSOLabels;
%rename(RealNumber) CRealNumber;
%rename(HMSVMModel) CHMSVMModel;
%rename(SequenceLabels) CSequenceLabels;
%rename(Sequence) CSequence;
%rename(StateModel) CStateModel;
%rename(TwoStateModel) CTwoStateModel;
%rename(DirectorStructuredModel) CDirectorStructuredModel;
%rename(MultilabelSOLabels) CMultilabelSOLabels;
%rename(SparseMultilabel) CSparseMultilabel;
%rename(MultilabelModel) CMultilabelModel;
%rename(HashedMultilabelModel) CHashedMultilabelModel;
%rename(MultilabelCLRModel) CMultilabelCLRModel;
%rename(HierarchicalMultilabelModel) CHierarchicalMultilabelModel;

%rename(FactorType) CFactorType;
%rename(TableFactorType) CTableFactorType;
%rename(FactorDataSource) CFactorDataSource;
%rename(Factor) CFactor;
%rename(DisjointSet) CDisjointSet;
%rename(FactorGraph) CFactorGraph;
%rename(FactorGraphObservation) CFactorGraphObservation;
%rename(FactorGraphLabels) CFactorGraphLabels;
%rename(FactorGraphFeatures) CFactorGraphFeatures;
%rename(MAPInference) CMAPInference;
%rename(GraphCut) CGraphCut;
%rename(FactorGraphModel) CFactorGraphModel;

%rename(SOSVMHelper) CSOSVMHelper;
%rename(StructuredOutputMachine) CStructuredOutputMachine;
%rename(LinearStructuredOutputMachine) CLinearStructuredOutputMachine;
%rename(KernelStructuredOutputMachine) CKernelStructuredOutputMachine;

#ifdef USE_GPL_SHOGUN
%rename(DualLibQPBMSOSVM) CDualLibQPBMSOSVM;
#ifdef USE_MOSEK
%rename(PrimalMosekSOSVM) CPrimalMosekSOSVM;
#endif /* USE_MOSEK */
#endif //USE_GPL_SHOGUN

%rename(StochasticSOSVM) CStochasticSOSVM;
%rename(FWSOSVM) CFWSOSVM;

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
%include <shogun/machine/KernelStructuredOutputMachine.h>

#ifdef USE_GPL_SHOGUN
%include <shogun/structure/DualLibQPBMSOSVM.h>

#ifdef USE_MOSEK
%include <shogun/structure/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */
#endif //USE_GPL_SHOGUN

%include <shogun/structure/StochasticSOSVM.h>
%include <shogun/structure/FWSOSVM.h>
