/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soumyajit De, Björn Esser
 */

#ifndef DENSE_MATRIX_EXACT_LOG_H_
#define DENSE_MATRIX_EXACT_LOG_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>


namespace shogun
{

template<class T> class SGVector;
template<class T> class CDenseMatrixOperator;

/** @brief Class that generates jobs for computing logarithm of
 *  a dense matrix linear operator
 */
class CDenseMatrixExactLog : public COperatorFunction<float64_t>
{
public:
	/** default constructor */
	CDenseMatrixExactLog();

	/**
	 * constructor
	 *
	 * @param op the dense matrix linear operator for this operator function
	 */
	CDenseMatrixExactLog(CDenseMatrixOperator<float64_t>* op);

	/** destructor */
	virtual ~CDenseMatrixExactLog();

	/**
	 * precompute method that computes the log of the linear operator using
	 * Eigen3, creates a new linear operator and sets that as the operator to
	 * apply to vectors for this operator function
	 */
	virtual void precompute();

	/**
	 * method that solves the result for a sample
	 */
	virtual float64_t solve(SGVector<float64_t> sample);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DenseMatrixExactLog";
	}
};

}

#endif // DENSE_MATRIX_EXACT_LOG_H_
