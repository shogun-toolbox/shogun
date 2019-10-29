/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Grigorii Guz, Heiko Strathmann, Sergey Lisitsyn, Viktor Gal
 */

/* Remove C Prefix */
%shared_ptr(shogun::Math)
%shared_ptr(shogun::Statistics)

/* Trace samplers */
%shared_ptr(shogun::TraceSampler)
%shared_ptr(shogun::Seedable<shogun::TraceSampler>)
%shared_ptr(shogun::RandomMixin<shogun::TraceSampler, std::mt19937_64>)
%shared_ptr(shogun::NormalSampler)
%shared_ptr(shogun::ProbingSampler)

#ifdef USE_FLOAT32
%shared_ptr(shagun::SparseMatrixOperator<float32_t>)
#endif

#ifdef USE_FLOAT64
%shared_ptr(shogun::LinearOperator<float64_t>)
%shared_ptr(shogun::MatrixOperator<float64_t>)
%shared_ptr(shogun::SparseMatrixOperator<float64_t>)
%shared_ptr(shogun::DenseMatrixOperator<float64_t>)
%shared_ptr(shogun::LinearSolver<float64_t, float64_t>)
%shared_ptr(shogun::IterativeLinearSolver<float64_t,float64_t>)
%shared_ptr(shogun::OperatorFunction<float64_t>)
#endif
#ifdef USE_COMPLEX128
%shared_ptr(shogun::LinearOperator<complex128_t>)
%shared_ptr(shogun::MatrixOperator<complex128_t>)
%shared_ptr(shogun::SparseMatrixOperator<complex128_t>)
%shared_ptr(shogun::DenseMatrixOperator<complex128_t>)
%shared_ptr(shogun::LinearSolver<complex128_t, float64_t>)
%shared_ptr(shogun::IterativeLinearSolver<complex128_t,float64_t>)
%shared_ptr(shogun::IterativeShiftedLinearFamilySolver<float64_t,complex128_t>)
#endif

#ifdef USE_GPL_SHOGUN
%shared_ptr(shogun::SparseInverseCovariance)
#endif //USE_GPL_SHOGUN

// fix overloaded methods in Math
#if defined(SWIGLUA) || defined(SWIGR)

namespace shogun
{
#ifdef USE_INT32
%rename(pow_int32) Math::pow(int32_t,int32_t);
#endif

#ifdef USE_FLOAT32
%rename(sqrt_float32) std::sqrt(float32_t);
#endif

#ifdef USE_FLOAT64
%rename(pow_float64_int32) Math::pow(float64_t,int32_t);
%rename(pow_float64_float64) Math::pow(float64_t,float64_t);
%rename(sqrt_float64) std::sqrt(float64_t);
}
#endif

#ifdef USE_COMPLEX128
/*%rename(shogun::pow_complex128_float64) Math::pow(complex128_t,float64_t)
%rename(shogun::pow_complex128_int32) Math::pow(complex128_t,int32_t)*/
#endif

#endif // defined(SWIGLUA) || defined(SWIGR)

/* Log-det framework */

/* Linear operators */
%include <shogun/mathematics/linalg/linop/LinearOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealLinearOperator) LinearOperator<float64_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexLinearOperator) LinearOperator<complex128_t>;
#endif
}

%include <shogun/mathematics/linalg/linop/MatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealMatrixOperator) MatrixOperator<float64_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexMatrixOperator) MatrixOperator<complex128_t>;
#endif
}

%include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealSparseMatrixOperator) SparseMatrixOperator<float64_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexSparseMatrixOperator) SparseMatrixOperator<complex128_t>;
#endif
}

%include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealDenseMatrixOperator) DenseMatrixOperator<float64_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexDenseMatrixOperator) DenseMatrixOperator<complex128_t>;
#endif
}

/* Operator functions */
%include <shogun/mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealOperatorFunction) OperatorFunction<float64_t>;
#endif
}

%shared_ptr(shogun::RationalApproximation)
%shared_ptr(shogun::LogRationalApproximationIndividual)
%shared_ptr(shogun::LogRationalApproximationCGM)

/* Linear solvers */
%include <shogun/mathematics/linalg/linsolver/LinearSolver.h>
namespace shogun
{
#if defined(USE_FLOAT64)
    %template(RealLinearSolver) LinearSolver<float64_t,float64_t>;
#endif
#if defined(USE_FLOAT64) && defined(USE_COMPLEX128)
    %template(ComplexRealLinearSolver) LinearSolver<complex128_t,float64_t>;
#endif
}

%shared_ptr(shogun::DirectSparseLinearSolver)
#ifdef USE_COMPLEX128
  %shared_ptr(shogun::DirectLinearSolverComplex)
#endif

%include <shogun/mathematics/linalg/linsolver/IterativeLinearSolver.h>
namespace shogun
{
#if defined(USE_FLOAT64)
    %template(RealIterativeLinearSolver) IterativeLinearSolver<float64_t,float64_t>;
#endif
#if defined(USE_FLOAT64) && defined(USE_COMPLEX128)
    %template(ComplexRealIterativeLinearSolver) IterativeLinearSolver<complex128_t,float64_t>;
#endif
}

%shared_ptr(shogun::ConjugateGradientSolver)
%shared_ptr(shogun::ConjugateOrthogonalCGSolver)

%include <shogun/mathematics/linalg/linsolver/IterativeShiftedLinearFamilySolver.h>
namespace shogun
{
#if defined(USE_FLOAT64) && defined(USE_COMPLEX128)
    %template(RealComplexIterativeShiftedLinearSolver) IterativeShiftedLinearFamilySolver<float64_t,complex128_t>;
#endif
}

%shared_ptr(shogun::CGMShiftedFamilySolver)

%shared_ptr(shogun::EigenSolver)
%shared_ptr(shogun::LanczosEigenSolver)

%shared_ptr(shogun::LogDetEstimator)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/mathematics/Math.h>
%include <shogun/mathematics/Statistics.h>
#ifdef USE_GPL_SHOGUN
%include <shogun/mathematics/SparseInverseCovariance.h>
#endif //USE_GPL_SHOGUN

/* Log-det framework */
%include <shogun/mathematics/linalg/ratapprox/tracesampler/TraceSampler.h>
namespace shogun
{
    %template(SeedableTraceSampler) Seedable<TraceSampler>;
    %template(RandomMixinTraceSampler) RandomMixin<TraceSampler, std::mt19937_64>;
}

%include <shogun/mathematics/linalg/ratapprox/tracesampler/NormalSampler.h>
%include <shogun/mathematics/linalg/ratapprox/tracesampler/ProbingSampler.h>

%include <shogun/mathematics/linalg/linop/LinearOperator.h>
%include <shogun/mathematics/linalg/linop/MatrixOperator.h>
%include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
%include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>

%include <shogun/mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>
%include <shogun/mathematics/linalg/ratapprox/opfunc/RationalApproximation.h>
%include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationIndividual.h>
%include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationCGM.h>

%include <shogun/mathematics/linalg/linsolver/LinearSolver.h>
%include <shogun/mathematics/linalg/linsolver/DirectSparseLinearSolver.h>
%include <shogun/mathematics/linalg/linsolver/DirectLinearSolverComplex.h>
%include <shogun/mathematics/linalg/linsolver/IterativeLinearSolver.h>
%include <shogun/mathematics/linalg/linsolver/ConjugateGradientSolver.h>
%include <shogun/mathematics/linalg/linsolver/ConjugateOrthogonalCGSolver.h>
%include <shogun/mathematics/linalg/linsolver/IterativeShiftedLinearFamilySolver.h>
%include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>

%include <shogun/mathematics/linalg/eigsolver/EigenSolver.h>
%include <shogun/mathematics/linalg/eigsolver/LanczosEigenSolver.h>

%include <shogun/mathematics/linalg/ratapprox/logdet/LogDetEstimator.h>
