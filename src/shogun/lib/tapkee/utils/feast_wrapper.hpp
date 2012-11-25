/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_FEAST_WRAPPER_H_
#define TAPKEE_FEAST_WRAPPER_H_

#include <complex>

namespace feast_internal
{
	template<typename Scalar> struct feast_wrapper;
}

template <class LMatrixType, class RMatrixType, class MatrixOperation>
class FeastGeneralizedSelfAdjointEigenSolver
{
	FeastGeneralizedSelfAdjointEigenSolver(const LMatrixType& A, const RMatrixType& B)
	{
		compute(A,B);
	}

	FeastGeneralizedSelfAdjointEigenSolver& compute(const LMatrixType& A, const RMatrixType& B);
};

template <class LMatrixType, class RMatrixType, class MatrixOperation>
FeastGeneralizedSelfAdjointEigenSolver<LMatrixType,RMatrixType>& 
	FeastGeneralizedSelfAdjointEigenSolver<LMatrixType,RMatrixType,MatrixOperation>
::compute(const LMatrixType& A, const RMatrixType& B)
{
	typedef Complex std::complex<Scalar>;

	int ijob = -1;
	int N = A.cols();
	
	Complex Ze;
	Scalar* work1 = new Scalar[N*MO];
	Complex* work2 = new Complex[N*MO]
	Scalar* Aq = new Scalar[MO*MO];
	Scalar* Bq = new Scalar[MO*MO];

	do
	{
		feast_internal::
		switch (ijob)
		{
			case 10:
				break;
			case 11:
				break;
			case 30:
				break;
			case 40:
				break;
		}
	}
	while (ijob != 0)
}

#endif 
