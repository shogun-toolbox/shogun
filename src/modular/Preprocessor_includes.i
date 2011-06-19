%{
#include <shogun/lib/Compressor.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/preprocessor/SparsePreprocessor.h>

#include <shogun/preprocessor/SimplePreprocessor.h>
#include <shogun/preprocessor/PCACut.h>
#include <shogun/preprocessor/KernelPCACut.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/preprocessor/LogPlusOne.h>
#include <shogun/preprocessor/PruneVarSubMean.h>
#include <shogun/preprocessor/NormDerivativeLem3.h>
#include <shogun/preprocessor/RandomFourierGaussPreproc.h>

#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/preprocessor/ClassicMDS.h>
#include <shogun/preprocessor/LocallyLinearEmbedding.h>
#include <shogun/preprocessor/Isomap.h>

#include <shogun/preprocessor/StringPreprocessor.h>
#include <shogun/preprocessor/DecompressString.h>
#include <shogun/preprocessor/SortUlongString.h>
#include <shogun/preprocessor/SortWordString.h>
%}
