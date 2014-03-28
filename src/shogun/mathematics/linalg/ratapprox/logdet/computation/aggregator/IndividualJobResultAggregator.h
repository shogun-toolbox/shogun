/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef INDIVIDUAL_JOB_RESULT_AGGREGATOR_H_
#define INDIVIDUAL_JOB_RESULT_AGGREGATOR_H_

#include <shogun/lib/config.h>
#include <shogun/lib/computation/aggregator/StoreVectorAggregator.h>

#ifdef HAVE_EIGEN3

namespace shogun
{
class CJobResult;
template<class T> class SGVector;
template<class RetType, class OperandType> class CLinearOperator;

/** @brief Class that aggregates vector job results in each submit_result call
 * of jobs generated from rational approximation of linear operator function
 * times a vector. finalize extracts the imaginary part of that aggregation,
 * applies the linear operator to the aggregation, performs a dot product with
 * the sample vector, multiplies with the constant multiplier (see
 * CRationalApproximation) and stores the result as CScalarResult
 */
class CIndividualJobResultAggregator : public CStoreVectorAggregator<complex128_t>
{
public:
	/** default constructor */
	CIndividualJobResultAggregator();

	/**
	 * constructor
	 *
	 * @param linear_operator linear operator to apply on the imaginary part
	 * of the aggregation before performing the dot product
	 * @param vector the sample vector with which the final dot product
	 * has to be performed
	 * @param const_multiplier the constant multiplier to be multiplied with
	 * the final vector-vector product to give final result
	 */
	CIndividualJobResultAggregator(CLinearOperator< SGVector<float64_t>, SGVector<float64_t> >*
		linear_operator, SGVector<float64_t> vector,
		const float64_t& const_multiplier);

	/** destructor */
	virtual ~CIndividualJobResultAggregator();

	/**
	 * method that finalizes the aggregation and computes the result (scalar),
	 * its necessary to call finalize before getting the final result
	 */
	virtual void finalize();

	/** @return object name */
	virtual const char* get_name() const
	{
		return "IndividualJobResultAggregator";
	}
private:
	/** the linear operator */
	CLinearOperator< SGVector<float64_t>, SGVector<float64_t> >* m_linear_operator;

	/** the sample vector */
	SGVector<float64_t> m_vector;

	/** the constant multiplier */
	const float64_t m_const_multiplier;

	/** initialize with default values and register params */
	void init();
};

}

#endif // HAVE_EIGEN3
#endif // INDIVIDUAL_JOB_RESULT_AGGREGATOR_H_
