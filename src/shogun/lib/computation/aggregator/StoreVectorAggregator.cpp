/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/SGVector.h>
#include <shogun/lib/computation/jobresult/VectorResult.h>
#include <shogun/lib/computation/aggregator/StoreVectorAggregator.h>
#include <shogun/lib/computation/aggregator/JobResultAggregator.h>

namespace shogun
{
template<class T>
CStoreVectorAggregator<T>::CStoreVectorAggregator()
	: CJobResultAggregator()
	{
		init();
		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

template<class T>
CStoreVectorAggregator<T>::CStoreVectorAggregator(index_t dimension)
	: CJobResultAggregator()
	{
		init();

		m_aggregate=SGVector<T>(dimension);
		m_aggregate.set_const(static_cast<T>(0));
	}

template<class T>
void CStoreVectorAggregator<T>::init()
	{
		SG_ADD(&m_aggregate, "current_aggregate",
			"Aggregation of computation job results", MS_NOT_AVAILABLE);
	}

template<class T>
CStoreVectorAggregator<T>::~CStoreVectorAggregator()
	{
	}

template<class T>
void CStoreVectorAggregator<T>::submit_result(CJobResult* result)
	{
		SG_GCDEBUG("Entering\n")

		// check for proper typecast
		CVectorResult<T>* new_result=dynamic_cast<CVectorResult<T>*>(result);
		if (!new_result)
			SG_ERROR("result is not of CVectorResult type!\n");
		// aggregate it with previous
		m_aggregate+=new_result->get_result();

		SG_GCDEBUG("Leaving\n")
	}

template<>
void CStoreVectorAggregator<bool>::submit_result(CJobResult* result)
	{
		SG_NOTIMPLEMENTED
	}

template<>
void CStoreVectorAggregator<char>::submit_result(CJobResult* result)
	{
		SG_NOTIMPLEMENTED
	}

template class CStoreVectorAggregator<bool>;
template class CStoreVectorAggregator<char>;
template class CStoreVectorAggregator<int8_t>;
template class CStoreVectorAggregator<uint8_t>;
template class CStoreVectorAggregator<int16_t>;
template class CStoreVectorAggregator<uint16_t>;
template class CStoreVectorAggregator<int32_t>;
template class CStoreVectorAggregator<uint32_t>;
template class CStoreVectorAggregator<int64_t>;
template class CStoreVectorAggregator<uint64_t>;
template class CStoreVectorAggregator<float32_t>;
template class CStoreVectorAggregator<float64_t>;
template class CStoreVectorAggregator<floatmax_t>;
template class CStoreVectorAggregator<complex128_t>;
}
