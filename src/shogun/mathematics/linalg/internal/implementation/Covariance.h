/*
 * Covariance.h
 *
 *  Created on: Mar 27, 2016
 *      Author: chris
 */

#ifndef SRC_SHOGUN_MATHEMATICS_LINALG_INTERNAL_IMPLEMENTATION_COVARIANCE_H_
#define SRC_SHOGUN_MATHEMATICS_LINALG_INTERNAL_IMPLEMENTATION_COVARIANCE_H_


#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

namespace shogun
{

namespace linalg
{

namespace implementation
{


/**
 * @Brief A generic class that has a compute method for finding
 * the covariance of a matrix.
 */
template<enum Backend, typename Matrix>
struct matrix_covariance{

	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Matrix return type */
	typedef SGMatrix<T> ReturnType;

	/** Computes the empirical estimate of the covariance matrix of the given
	 * data which is organized as num_cols variables with num_rows observations.
	 *
	 * Data is centered before matrix is computed. May be done in place.
	 * In this case, the observation matrix is changed (centered).
	 *
	 * Given sample matrix \f$X\f$, first, column mean i, bool in_place=falses removed to create
	 * \f$\bar X\f$. Then \f$\text{cov}(X)=(X-\bar X)^T(X - \bar X)\f$ is
	 * returned.
	 *
	 * @param m data matrix organized as one variable per column
	 * @param in_place optional, if set to true, observations matrix will be
	 * centered, if false, a copy will be created an centered.
	 * @return covariance matrix empirical estimate
	 */
	static ReturnType compute(Matrix m, bool in_place = false);

};

/**
 * @Brief A specialization of matrix_covariance that uses SGMatrix as its types
 * and uses Eigen3 as it's backend component
 */
template<typename Matrix>
struct matrix_covariance<Backend::EIGEN3, Matrix>{

	/** Generic scalar type */
	typedef typename Matrix::Scalar T;

	/** Matrix return type */
	typedef SGMatrix<T> ReturnType;

	/** Computes the empirical estimate of the covariance matrix of the given
	 * data which is organized as num_cols variables with num_rows observations.
	 *
	 * Data is centered before matrix is computed. May be done in place.
	 * In this case, the observation matrix is changed (centered).
	 *
	 * Given sample matrix \f$X\f$, first, column mean is removed to create
	 * \f$\bar X\f$. Then \f$\text{cov}(X)=(X-\bar X)^T(X - \bar X)\f$ is
	 * returned.
	 *
	 * @param observations data matrix organized as one variable per column
	 * @param in_place optional, if set to true, observations matrix will be
	 * centered, if false, a copy will be created an centered.
	 * @return covariance matrix empirical estimate
	 */
	static ReturnType compute(SGMatrix<T> observations, bool in_place = false){

		int32_t N = observations.num_rows;
		int32_t D = observations.num_cols;

		REQUIRE(N>1, "Number of observations (%d) must be at least 2.\n", N);
		REQUIRE(D>0, "Number of dimensions (%d) must be at least 1.\n", D);

		/* center observations, potentially in-place */
		SGMatrix<float64_t> centered;
		if (!in_place)
		{
			centered = observations.clone();
		}
		else
			centered = observations;

		Map<MatrixXd> eigen_centered(centered.matrix, N, D);
		eigen_centered.rowwise() -= eigen_centered.colwise().mean();

		/* compute and store 1/(N-1) * X.T * X */
		SGMatrix<float64_t> cov(D, D);
		Map<MatrixXd> eigen_cov(cov.matrix, D, D);
		eigen_cov = (eigen_centered.adjoint() * eigen_centered) / double(N);

		return cov;
	}
};

}

}

}


#endif /* SRC_SHOGUN_MATHEMATICS_LINALG_INTERNAL_IMPLEMENTATION_COVARIANCE_H_ */
