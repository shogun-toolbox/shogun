/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Written (W) 2013 Heiko Strathmann
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

/* Remove C Prefix */
%rename(Math) CMath;
%rename(Statistics) CStatistics;
#ifdef USE_GPL_SHOGUN
%rename(SparseInverseCovariance) CSparseInverseCovariance;
#endif //USE_GPL_SHOGUN

// fix overloaded methods in Math
#if defined(SWIGLUA) || defined(SWIGR)

namespace shogun
{
#ifdef USE_INT32
%rename(pow_int32) CMath::pow(int32_t,int32_t);
%rename(random_int32) CMath::random(int32_t,int32_t);
#endif

#ifdef USE_UINT32
%rename(random_uint32) CMath::random(uint32_t,uint32_t);
#endif

#ifdef USE_INT64
%rename(random_int64) CMath::random(int64_t,int64_t);
#endif

#ifdef USE_UINT64
%rename(random_uint64) CMath::random(uint64_t,uint64_t);
#endif

#ifdef USE_FLOAT32
%rename(normal_random_float32) CMath::normal_random(float32_t,float32_t);
%rename(random_float32) CMath::random(float32_t,float32_t);
%rename(sqrt_float32) CMath::sqrt(float32_t);
#endif

#ifdef USE_FLOAT64
%rename(normal_random_float64) CMath::normal_random(float64_t,float64_t);
%rename(pow_float64_int32) CMath::pow(float64_t,int32_t);
%rename(pow_float64_float64) CMath::pow(float64_t,float64_t);
%rename(random_float64) CMath::random(float64_t,float64_t);
%rename(sqrt_float64) CMath::sqrt(float64_t);
}
#endif

#ifdef USE_COMPLEX128
%rename(pow_complex128_float64) CMath::pow(complex128_t,float64_t);
%rename(pow_complex128_int32) CMath::pow(complex128_t,int32_t);
#endif

#endif // defined(SWIGLUA) || defined(SWIGR)

/* Log-det framework */

/* Trace samplers */
%rename(TraceSampler) CTraceSampler;
%rename(NormalSampler) CNormalSampler;
%rename(ProbingSampler) CProbingSampler;

/* Linear operators */
%include <shogun/mathematics/linalg/linop/LinearOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealLinearOperator) CLinearOperator<float64_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexLinearOperator) CLinearOperator<complex128_t>;
#endif
}

%include <shogun/mathematics/linalg/linop/MatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealMatrixOperator) CMatrixOperator<float64_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexMatrixOperator) CMatrixOperator<complex128_t>;
#endif
}

%include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealSparseMatrixOperator) CSparseMatrixOperator<float64_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexSparseMatrixOperator) CSparseMatrixOperator<complex128_t>;
#endif
}

%include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealDenseMatrixOperator) CDenseMatrixOperator<float64_t>;
#endif
#ifdef USE_COMPLEX128
    %template(ComplexDenseMatrixOperator) CDenseMatrixOperator<complex128_t>;
#endif
}

/* Operator functions */
%include <shogun/mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealOperatorFunction) COperatorFunction<float64_t>;
#endif
}

%rename(RationalApproximation) CRationalApproximation;
%rename(LogRationalApproximationIndividual) CLogRationalApproximationIndividual;
%rename(LogRationalApproximationCGM) CLogRationalApproximationCGM;

/* Linear solvers */
%include <shogun/mathematics/linalg/linsolver/LinearSolver.h>
namespace shogun
{
#if defined(USE_FLOAT64)
    %template(RealLinearSolver) CLinearSolver<float64_t,float64_t>;
#endif
#if defined(USE_FLOAT64) && defined(USE_COMPLEX128)
    %template(ComplexRealLinearSolver) CLinearSolver<complex128_t,float64_t>;
#endif
}

%rename(DirectSparseLinearSolver) CDirectSparseLinearSolver;
#ifdef USE_COMPLEX128
  %rename(DirectLinearSolverComplex) CDirectLinearSolverComplex;
#endif

%include <shogun/mathematics/linalg/linsolver/IterativeLinearSolver.h>
namespace shogun
{
#if defined(USE_FLOAT64)
    %template(RealIterativeLinearSolver) CIterativeLinearSolver<float64_t,float64_t>;
#endif
#if defined(USE_FLOAT64) && defined(USE_COMPLEX128)
    %template(ComplexRealIterativeLinearSolver) CIterativeLinearSolver<complex128_t,float64_t>;
#endif
}

%rename (ConjugateGradientSolver) CConjugateGradientSolver;
%rename (ConjugateOrthogonalCGSolver) CConjugateOrthogonalCGSolver;

%include <shogun/mathematics/linalg/linsolver/IterativeShiftedLinearFamilySolver.h>
namespace shogun
{
#if defined(USE_FLOAT64) && defined(USE_COMPLEX128)
    %template(RealComplexIterativeShiftedLinearSolver) CIterativeShiftedLinearFamilySolver<float64_t,complex128_t>;
#endif
}

%rename(CGMShiftedFamilySolver) CCGMShiftedFamilySolver;

%rename(EigenSolver) CEigenSolver;
%rename(LanczosEigenSolver) CLanczosEigenSolver;

%rename(LogDetEstimator) CLogDetEstimator;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/mathematics/Math.h>
%include <shogun/mathematics/Statistics.h>
#ifdef USE_GPL_SHOGUN
%include <shogun/mathematics/SparseInverseCovariance.h>
#endif //USE_GPL_SHOGUN

/* Log-det framework */
%include <shogun/mathematics/linalg/ratapprox/tracesampler/TraceSampler.h>
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
