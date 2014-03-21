/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef STORE_VECTOR_AGGREGATOR_H_
#define STORE_VECTOR_AGGREGATOR_H_

#include <shogun/lib/config.h>

#include <shogun/lib/computation/aggregator/JobResultAggregator.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>

namespace shogun
{
class CJobResult;
template <class T> class CVectorResult;

/** @brief Abstract template class that aggregates vector job results in each
 * submit_result call, finalize is abstract
 */
template<class T> class CStoreVectorAggregator : public CJobResultAggregator
{
public:
	/** default constructor */
	CStoreVectorAggregator();

	/**
	 * constructor
	 *
	 * @param dimension the dimension of vectors in vector result
	 */
	CStoreVectorAggregator(index_t dimension);

	/** destructor */
	virtual ~CStoreVectorAggregator();

	/**
	 * method that submits the result (vector) of an independent job, and
	 * computes the aggregation with the previously submitted result
	 *
	 * @param result the result of an independent job
	 */
	virtual void submit_result(CJobResult* result);

	/**
	 * abstract method that finalizes the aggregation and computes the result
	 * (scalar), its necessary to call finalize before getting the final result
	 */
	virtual void finalize() = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "StoreVectorAggregator";
	}
protected:
	/** the aggregation */
	SGVector<T> m_aggregate;

private:
	/** initialize with default values and register params */
	void init();
};

}

#endif // STORE_VECTOR_AGGREGATOR_H_
