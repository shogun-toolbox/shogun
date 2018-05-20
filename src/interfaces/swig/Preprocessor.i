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

%rename(PCA) CPCA;
%rename(KernelPCA) CKernelPCA;
%rename(FisherLda) CFisherLDA;

%rename(SortUlongString) CSortUlongString;
%rename(SortWordString) CSortWordString;

/* Feature selection framework */
#%rename(DependenceMaximization) CDependenceMaximization;
#%rename(KernelDependenceMaximization) CDependenceMaximization;

%newobject shogun::CFeatureSelection::apply;
%newobject shogun::CFeatureSelection::remove_feats;

%newobject shogun::CKernelPCA::apply_to_string_features;

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
