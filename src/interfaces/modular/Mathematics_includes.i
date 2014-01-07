%{
#include <mathematics/Math.h>
#include <mathematics/Statistics.h>
#include <mathematics/SparseInverseCovariance.h>

/* Log-det framework */
#include <mathematics/linalg/ratapprox/tracesampler/TraceSampler.h>
#include <mathematics/linalg/ratapprox/tracesampler/NormalSampler.h>
#include <mathematics/linalg/ratapprox/tracesampler/ProbingSampler.h>

#include <mathematics/linalg/linop/LinearOperator.h>
#include <mathematics/linalg/linop/MatrixOperator.h>
#include <mathematics/linalg/linop/SparseMatrixOperator.h>
#include <mathematics/linalg/linop/DenseMatrixOperator.h>

#include <mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>
#include <mathematics/linalg/ratapprox/opfunc/RationalApproximation.h>
#include <mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationIndividual.h>
#include <mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationCGM.h>

#include <mathematics/linalg/linsolver/LinearSolver.h>
#include <mathematics/linalg/linsolver/DirectSparseLinearSolver.h>
#include <mathematics/linalg/linsolver/DirectLinearSolverComplex.h>
#include <mathematics/linalg/linsolver/IterativeLinearSolver.h>
#include <mathematics/linalg/linsolver/ConjugateGradientSolver.h>
#include <mathematics/linalg/linsolver/ConjugateOrthogonalCGSolver.h>
#include <mathematics/linalg/linsolver/IterativeShiftedLinearFamilySolver.h>
#include <mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>

#include <mathematics/linalg/eigsolver/EigenSolver.h>
#include <mathematics/linalg/eigsolver/LanczosEigenSolver.h>

#include <mathematics/linalg/ratapprox/logdet/LogDetEstimator.h>
%}
