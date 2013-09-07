%{
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/SparseInverseCovariance.h>

/* Log-det framework */
#include <shogun/mathematics/logdet/TraceSampler.h>
#include <shogun/mathematics/logdet/NormalSampler.h>
#include <shogun/mathematics/logdet/ProbingSampler.h>

#include <shogun/mathematics/logdet/LinearOperator.h>
#include <shogun/mathematics/logdet/MatrixOperator.h>
#include <shogun/mathematics/logdet/SparseMatrixOperator.h>

#include <shogun/mathematics/logdet/OperatorFunction.h>
#include <shogun/mathematics/logdet/RationalApproximation.h>
#include <shogun/mathematics/logdet/LogRationalApproximationCGM.h>

#include <shogun/mathematics/logdet/LinearSolver.h>
#include <shogun/mathematics/logdet/DirectSparseLinearSolver.h>
#include <shogun/mathematics/logdet/IterativeLinearSolver.h>
#include <shogun/mathematics/logdet/IterativeShiftedLinearFamilySolver.h>
#include <shogun/mathematics/logdet/CGMShiftedFamilySolver.h>

#include <shogun/mathematics/logdet/EigenSolver.h>
#include <shogun/mathematics/logdet/LanczosEigenSolver.h>

#include <shogun/mathematics/logdet/LogDetEstimator.h>
%}