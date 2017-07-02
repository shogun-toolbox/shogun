%{
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/SparseInverseCovariance.h>

/* Log-det framework */
#include <shogun/mathematics/linalg/ratapprox/tracesampler/TraceSampler.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/NormalSampler.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/ProbingSampler.h>

#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/linop/MatrixOperator.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>

#include <shogun/mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>
#include <shogun/mathematics/linalg/ratapprox/opfunc/RationalApproximation.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationIndividual.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationCGM.h>

#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>
#include <shogun/mathematics/linalg/linsolver/DirectSparseLinearSolver.h>
#include <shogun/mathematics/linalg/linsolver/DirectLinearSolverComplex.h>
#include <shogun/mathematics/linalg/linsolver/IterativeLinearSolver.h>
#include <shogun/mathematics/linalg/linsolver/ConjugateGradientSolver.h>
#include <shogun/mathematics/linalg/linsolver/ConjugateOrthogonalCGSolver.h>
#include <shogun/mathematics/linalg/linsolver/IterativeShiftedLinearFamilySolver.h>
#include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>

#include <shogun/mathematics/linalg/eigsolver/EigenSolver.h>
#include <shogun/mathematics/linalg/eigsolver/LanczosEigenSolver.h>

#include <shogun/mathematics/linalg/ratapprox/logdet/LogDetEstimator.h>
%}
