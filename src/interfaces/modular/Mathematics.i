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
%rename(SparseInverseCovariance) CSparseInverseCovariance;

/* Log-det framework */

#ifdef HAVE_EIGEN3
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
#ifdef USE_COMPLEX64
    %template(ComplexLinearOperator) CLinearOperator<complex64_t>;
#endif
}

%include <shogun/mathematics/linalg/linop/MatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealMatrixOperator) CMatrixOperator<float64_t>;
#endif
#ifdef USE_COMPLEX64
    %template(ComplexMatrixOperator) CMatrixOperator<complex64_t>;
#endif
}

%include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealSparseMatrixOperator) CSparseMatrixOperator<float64_t>;
#endif
#ifdef USE_COMPLEX64
    %template(ComplexSparseMatrixOperator) CSparseMatrixOperator<complex64_t>;
#endif
}

%include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealDenseMatrixOperator) CDenseMatrixOperator<float64_t>;
#endif
#ifdef USE_COMPLEX64
    %template(ComplexDenseMatrixOperator) CDenseMatrixOperator<complex64_t>;
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
#if defined(USE_FLOAT64) && defined(USE_COMPLEX64)
    %template(ComplexRealLinearSolver) CLinearSolver<complex64_t,float64_t>;
#endif
}

%rename(DirectSparseLinearSolver) CDirectSparseLinearSolver;
#ifdef USE_COMPLEX64
  %rename(DirectLinearSolverComplex) CDirectLinearSolverComplex;
#endif

%include <shogun/mathematics/linalg/linsolver/IterativeLinearSolver.h>
namespace shogun
{
#if defined(USE_FLOAT64)
    %template(RealIterativeLinearSolver) CIterativeLinearSolver<float64_t,float64_t>;
#endif
#if defined(USE_FLOAT64) && defined(USE_COMPLEX64)
    %template(ComplexRealIterativeLinearSolver) CIterativeLinearSolver<complex64_t,float64_t>;
#endif
}

%rename (ConjugateGradientSolver) CConjugateGradientSolver;
%rename (ConjugateOrthogonalCGSolver) CConjugateOrthogonalCGSolver;

%include <shogun/mathematics/linalg/linsolver/IterativeShiftedLinearFamilySolver.h>
namespace shogun
{
#if defined(USE_FLOAT64) && defined(USE_COMPLEX64)
    %template(RealComplexIterativeShiftedLinearSolver) CIterativeShiftedLinearFamilySolver<float64_t,complex64_t>;
#endif
}

%rename(CGMShiftedFamilySolver) CCGMShiftedFamilySolver;

%rename(EigenSolver) CEigenSolver;
%rename(LanczosEigenSolver) CLanczosEigenSolver;

%rename(LogDetEstimator) CLogDetEstimator;
#endif // HAVE_EIGEN3

/* Include Class Headers to make them visible from within the target language */
%include <shogun/mathematics/Math.h>
%include <shogun/mathematics/Statistics.h>
%include <shogun/mathematics/SparseInverseCovariance.h>

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
