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
%rename(TraceSampler) CTraceSampler;
%rename(NormalSampler) CNormalSampler;
%rename(ProbingSampler) CProbingSampler;

%include <shogun/mathematics/logdet/LinearOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealLinearOperator) CLinearOperator<float64_t>;
#endif
}

%include <shogun/mathematics/logdet/MatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealMatrixOperator) CMatrixOperator<float64_t>;
#endif
}

%include <shogun/mathematics/logdet/SparseMatrixOperator.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealSparseMatrixOperator) CSparseMatrixOperator<float64_t>;
#endif
}

%include <shogun/mathematics/logdet/OperatorFunction.h>
namespace shogun
{
#ifdef USE_FLOAT64
    %template(RealOperatorFunction) COperatorFunction<float64_t>;
#endif
}

%rename(RationalApproximation) CRationalApproximation;
%rename(LogRationalApproximationCGM) CLogRationalApproximationCGM;

%include <shogun/mathematics/logdet/LinearSolver.h>
namespace shogun
{
#if defined(USE_FLOAT64)
    %template(RealLinearSolver) CLinearSolver<float64_t,float64_t>;
#endif
}

%rename(DirectSparseLinearSolver) CDirectSparseLinearSolver;

%include <shogun/mathematics/logdet/IterativeLinearSolver.h>
namespace shogun
{
#if defined(USE_FLOAT64)
    %template(RealIterativeLinearSolver) CIterativeLinearSolver<float64_t,float64_t>;
#endif
}

%include <shogun/mathematics/logdet/IterativeShiftedLinearFamilySolver.h>
namespace shogun
{
#if defined(USE_FLOAT64) //TODO check complex 64 here && defined(USE_COMPLEX64)
    %template(RealComplexIterativeShiftedLinearSolver) CIterativeShiftedLinearFamilySolver<float64_t,complex64_t>;
#endif
}

%include <shogun/mathematics/logdet/CGMShiftedFamilySolver.h>

%include <shogun/mathematics/logdet/RationalApproximation.h>
%include <shogun/mathematics/logdet/LogRationalApproximationCGM.h>

%rename(EigenSolver) CEigenSolver;
%rename(LanczosEigenSolver) CLanczosEigenSolver;

%rename(LogDetEstimator) CLogDetEstimator;
#endif

/* Include Class Headers to make them visible from within the target language */
%include <shogun/mathematics/Math.h>
%include <shogun/mathematics/Statistics.h>
%include <shogun/mathematics/SparseInverseCovariance.h>

/* Log-det framework */
%include <shogun/mathematics/logdet/TraceSampler.h>
%include <shogun/mathematics/logdet/NormalSampler.h>
%include <shogun/mathematics/logdet/ProbingSampler.h>

%include <shogun/mathematics/logdet/LinearOperator.h>
%include <shogun/mathematics/logdet/MatrixOperator.h>
%include <shogun/mathematics/logdet/SparseMatrixOperator.h>

%include <shogun/mathematics/logdet/OperatorFunction.h>
%include <shogun/mathematics/logdet/RationalApproximation.h>
%include <shogun/mathematics/logdet/LogRationalApproximationCGM.h>

%include <shogun/mathematics/logdet/LinearSolver.h>
%include <shogun/mathematics/logdet/DirectSparseLinearSolver.h>
%include <shogun/mathematics/logdet/IterativeLinearSolver.h>
%include <shogun/mathematics/logdet/IterativeShiftedLinearFamilySolver.h>
%include <shogun/mathematics/logdet/CGMShiftedFamilySolver.h>

%include <shogun/mathematics/logdet/EigenSolver.h>
%include <shogun/mathematics/logdet/LanczosEigenSolver.h>

%include <shogun/mathematics/logdet/LogDetEstimator.h>