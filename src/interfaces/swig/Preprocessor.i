/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
%rename(Preprocessor) CPreprocessor;
%rename(SparsePreprocessor) CSparsePreprocessor;
%rename(SumOne) CSumOne;
%rename(NormOne) CNormOne;
%rename(LogPlusOne) CLogPlusOne;
%rename(PruneVarSubMean) CPruneVarSubMean;
%rename(RandomFourierGaussPreproc) CRandomFourierGaussPreproc;
%rename(HomogeneousKernelMap) CHomogeneousKernelMap;
%rename(PNorm) CPNorm;
%rename(RescaleFeatures) CRescaleFeatures;

%rename(DimensionReductionPreprocessor) CDimensionReductionPreprocessor;
%rename(PCA) CPCA;
%rename(KernelPCA) CKernelPCA;
%rename(FisherLda) CFisherLDA;

%rename(SortUlongString) CSortUlongString;
%rename(SortWordString) CSortWordString;

/* Feature selection framework */
#%rename(DependenceMaximization) CDependenceMaximization;
#%rename(KernelDependenceMaximization) CDependenceMaximization;

%rename(EmbeddingConverter) CEmbeddingConverter;
#ifdef USE_GPL_SHOGUN
%rename(NeighborhoodPreservingEmbedding) CNeighborhoodPreservingEmbedding;
%rename(LocalTangentSpaceAlignment) CLocalTangentSpaceAlignment;
%rename(LinearLocalTangentSpaceAlignment) CLinearLocalTangentSpaceAlignment;
%rename(HessianLocallyLinearEmbedding) CHessianLocallyLinearEmbedding;
%rename(LocallyLinearEmbedding) CLocallyLinearEmbedding;
%rename(KernelLocallyLinearEmbedding) CKernelLocallyLinearEmbedding;
%rename(LaplacianEigenmaps) CLaplacianEigenmaps;
%rename(StochasticProximityEmbedding) CStochasticProximityEmbedding;
#endif //USE_GPL_SHOGUN
%rename(LocalityPreservingProjections) CLocalityPreservingProjections;
%rename(MultidimensionalScaling) CMultidimensionalScaling;
%rename(Isomap) CIsomap;
%rename(DiffusionMaps) CDiffusionMaps;
%rename(FactorAnalysis) CFactorAnalysis;
%rename(TDistributedStochasticNeighborEmbedding) CTDistributedStochasticNeighborEmbedding;
%rename(ManifoldSculpting) CManifoldSculpting;
%rename(HashedDocConverter) CHashedDocConverter;
%rename(ICAConverter) CICAConverter;
%rename(Jade) CJade;
%rename(SOBI) CSOBI;
%rename(FFSep) CFFSep;
%rename(JediSep) CJediSep;
%rename(UWedgeSep) CUWedgeSep;
%rename(FastICA) CFastICA;

%newobject shogun::CFeatureSelection::apply;
%newobject shogun::CFeatureSelection::remove_feats;

%newobject shogun::CKernelPCA::apply_to_string_features;

%newobject shogun::CEmbeddingConverter::apply;
%newobject shogun::*::embed_kernel;
%newobject shogun::*::embed_distance;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/lib/Compressor.h>
%include <shogun/preprocessor/Preprocessor.h>

/* Templates Class DensePreprocessor*/
%include <shogun/preprocessor/DensePreprocessor.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealPreprocessor) CDensePreprocessor<float64_t>;
#endif
#ifdef USE_UINT64
    %template(UlongPreprocessor) CDensePreprocessor<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(WordPreprocessor) CDensePreprocessor<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortPreprocessor) CDensePreprocessor<int16_t>;
#endif
#ifdef USE_UINT8
    %template(BytePreprocessor) CDensePreprocessor<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(CharPreprocessor) CDensePreprocessor<char>;
#endif
}

/* Templates Class DimensionReductionPreprocessor */
%include <shogun/preprocessor/DimensionReductionPreprocessor.h>

/* Templates Class StringPreprocessor*/
%include <shogun/preprocessor/StringPreprocessor.h>
namespace shogun
{
#ifdef USE_UINT64
    %template(StringUlongPreprocessor) CStringPreprocessor<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(StringWordPreprocessor) CStringPreprocessor<uint16_t>;
#endif
#ifdef USE_UINT8
    %template(StringBytePreprocessor) CStringPreprocessor<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(StringCharPreprocessor) CStringPreprocessor<char>;
#endif
}

/* Templates Class DecompressString*/
%include <shogun/preprocessor/DecompressString.h>
namespace shogun
{
#ifdef USE_UINT64
    %template(DecompressUlongString) CDecompressString<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(DecompressWordString) CDecompressString<uint16_t>;
#endif
#ifdef USE_UINT8
    %template(DecompressByteString) CDecompressString<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(DecompressCharString) CDecompressString<char>;
#endif
}

/* Templates Class FeatureSelection */
%include <shogun/preprocessor/FeatureSelection.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealFeatureSelection) CFeatureSelection<float64_t>;
#endif
#ifdef USE_UINT64
    %template(UlongFeatureSelection) CFeatureSelection<uint64_t>;
#endif
#ifdef USE_UINT16
    %template(WordFeatureSelection) CFeatureSelection<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortFeatureSelection) CFeatureSelection<int16_t>;
#endif
#ifdef USE_UINT8
    %template(ByteFeatureSelection) CFeatureSelection<uint8_t>;
#endif
#ifdef USE_CHAR
    %template(CharFeatureSelection) CFeatureSelection<char>;
#endif
}

%include <shogun/preprocessor/SparsePreprocessor.h>
%include <shogun/preprocessor/NormOne.h>
%include <shogun/preprocessor/SumOne.h>
%include <shogun/preprocessor/LogPlusOne.h>
%include <shogun/preprocessor/PruneVarSubMean.h>
%include <shogun/preprocessor/RandomFourierGaussPreproc.h>
%include <shogun/preprocessor/HomogeneousKernelMap.h>
%include <shogun/preprocessor/PNorm.h>
%include <shogun/preprocessor/RescaleFeatures.h>

%include <shogun/preprocessor/PCA.h>
%include <shogun/preprocessor/KernelPCA.h>
%include <shogun/preprocessor/FisherLDA.h>

%include <shogun/preprocessor/SortUlongString.h>
%include <shogun/preprocessor/SortWordString.h>

%include <shogun/preprocessor/EmbeddingConverter.h>
#ifdef USE_GPL_SHOGUN
%include <shogun/preprocessor/LocallyLinearEmbedding.h>
%include <shogun/preprocessor/NeighborhoodPreservingEmbedding.h>
%include <shogun/preprocessor/LocalTangentSpaceAlignment.h>
%include <shogun/preprocessor/LinearLocalTangentSpaceAlignment.h>
%include <shogun/preprocessor/HessianLocallyLinearEmbedding.h>
%include <shogun/preprocessor/KernelLocallyLinearEmbedding.h>
%include <shogun/preprocessor/StochasticProximityEmbedding.h>
#endif //USE_GPL_SHOGUN
%include <shogun/preprocessor/LaplacianEigenmaps.h>
%include <shogun/preprocessor/LocalityPreservingProjections.h>
%include <shogun/preprocessor/MultidimensionalScaling.h>
%include <shogun/preprocessor/Isomap.h>
%include <shogun/preprocessor/DiffusionMaps.h>
%include <shogun/preprocessor/FactorAnalysis.h>
%include <shogun/preprocessor/TDistributedStochasticNeighborEmbedding.h>
%include <shogun/preprocessor/ManifoldSculpting.h>
%include <shogun/preprocessor/HashedDocConverter.h>
%include <shogun/preprocessor/ica/ICAConverter.h>
%include <shogun/preprocessor/ica/Jade.h>
%include <shogun/preprocessor/ica/SOBI.h>
%include <shogun/preprocessor/ica/FFSep.h>
%include <shogun/preprocessor/ica/JediSep.h>
%include <shogun/preprocessor/ica/UWedgeSep.h>
%include <shogun/preprocessor/ica/FastICA.h>
