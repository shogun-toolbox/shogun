%{
#include <shogun/lib/Compressor.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/preprocessor/SparsePreprocessor.h>

#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/preprocessor/SumOne.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/preprocessor/LogPlusOne.h>
#include <shogun/preprocessor/PruneVarSubMean.h>
#include <shogun/preprocessor/RandomFourierGaussPreproc.h>
#include <shogun/preprocessor/HomogeneousKernelMap.h>
#include <shogun/preprocessor/PNorm.h>
#include <shogun/preprocessor/RescaleFeatures.h>

#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/preprocessor/PCA.h>
#include <shogun/preprocessor/KernelPCA.h>
#include <shogun/preprocessor/FisherLDA.h>

#include <shogun/preprocessor/StringPreprocessor.h>
#include <shogun/preprocessor/DecompressString.h>
#include <shogun/preprocessor/SortUlongString.h>
#include <shogun/preprocessor/SortWordString.h>

#include <shogun/preprocessor/FeatureSelection.h>

#include <shogun/preprocessor/EmbeddingConverter.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/preprocessor/LocallyLinearEmbedding.h>
#include <shogun/preprocessor/NeighborhoodPreservingEmbedding.h>
#include <shogun/preprocessor/LocalTangentSpaceAlignment.h>
#include <shogun/preprocessor/LinearLocalTangentSpaceAlignment.h>
#include <shogun/preprocessor/HessianLocallyLinearEmbedding.h>
#include <shogun/preprocessor/KernelLocallyLinearEmbedding.h>
#include <shogun/preprocessor/StochasticProximityEmbedding.h>
#endif //USE_GPL_SHOGUN
#include <shogun/preprocessor/LaplacianEigenmaps.h>
#include <shogun/preprocessor/LocalityPreservingProjections.h>
#include <shogun/preprocessor/MultidimensionalScaling.h>
#include <shogun/preprocessor/Isomap.h>
#include <shogun/preprocessor/DiffusionMaps.h>
#include <shogun/preprocessor/FactorAnalysis.h>
#include <shogun/preprocessor/TDistributedStochasticNeighborEmbedding.h>
#include <shogun/preprocessor/ManifoldSculpting.h>
#include <shogun/preprocessor/HashedDocConverter.h>
#include <shogun/preprocessor/ica/ICAConverter.h>
#include <shogun/preprocessor/ica/Jade.h>
#include <shogun/preprocessor/ica/SOBI.h>
#include <shogun/preprocessor/ica/FFSep.h>
#include <shogun/preprocessor/ica/JediSep.h>
#include <shogun/preprocessor/ica/UWedgeSep.h>
#include <shogun/preprocessor/ica/FastICA.h>
%}
