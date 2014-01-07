%{
#include <lib/Compressor.h>
#include <features/FeatureTypes.h>
#include <preprocessor/Preprocessor.h>
#include <preprocessor/SparsePreprocessor.h>

#include <preprocessor/DensePreprocessor.h>
#include <preprocessor/SumOne.h>
#include <preprocessor/NormOne.h>
#include <preprocessor/LogPlusOne.h>
#include <preprocessor/PruneVarSubMean.h>
#include <preprocessor/RandomFourierGaussPreproc.h>
#include <preprocessor/HomogeneousKernelMap.h>
#include <preprocessor/PNorm.h>
#include <preprocessor/RescaleFeatures.h>

#include <preprocessor/DimensionReductionPreprocessor.h>
#include <preprocessor/PCA.h>
#include <preprocessor/KernelPCA.h>

#include <preprocessor/StringPreprocessor.h>
#include <preprocessor/DecompressString.h>
#include <preprocessor/SortUlongString.h>
#include <preprocessor/SortWordString.h>
%}
