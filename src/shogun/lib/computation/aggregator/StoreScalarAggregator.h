/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Thoralf Klein, Yuyu Zhang, Björn Esser
 */

#ifndef STORE_SCALAR_AGGREGATOR_H_
#define STORE_SCALAR_AGGREGATOR_H_

#include <shogun/lib/config.h>

#include <shogun/lib/computation/aggregator/JobResultAggregator.h>

namespace shogun
{
class CJobResult;

/** @brief Template class that aggregates scalar job results in each
 * submit_result call, finalize then transforms current aggregation into
 * a CScalarResult.
 */
template<class T> class CStoreScalarAggregator : public CJobResultAggregator
{
/** this class supports complex */
typedef bool supports_complex128_t;

public:
	/** default constructor */
	CStoreScalarAggregator();

	/** destructor */
	virtual ~CStoreScalarAggregator();

	/**
	 * method that submits the result (scalar) of an independent job, and
	 * computes the aggregation with the previously submitted result
	 *
	 * @param result the result of an independent job
	 */
	virtual void submit_result(CJobResult* result);

	/**
	 * method that finalizes the aggregation and computes the result (scalar),
	 * its necessary to call finalize before getting the final result
	 */
	virtual void finalize();

	/** @return object name */
	virtual const char* get_name() const
	{
		return "StoreScalarAggregator";
	}
private:
	/** the aggregation */
	T m_aggregate;

	/** initialize with default values and register params */
	void init();
};

}

#endif // STORE_SCALAR_AGGREGATOR_H_
