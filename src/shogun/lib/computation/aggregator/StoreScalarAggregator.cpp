/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>
#include <shogun/lib/computation/jobresult/ScalarResult.h>
#include <shogun/lib/computation/aggregator/StoreScalarAggregator.h>

namespace shogun
{
template<class T>
CStoreScalarAggregator<T>::CStoreScalarAggregator()
	: CJobResultAggregator()
	{
		init();
	}

template<class T>
void CStoreScalarAggregator<T>::init()
	{
		m_aggregate=static_cast<T>(0);

		set_generic<T>();

		SG_ADD(&m_aggregate, "current_aggregate",
			"Aggregation of computation job results", MS_NOT_AVAILABLE);
	}

template<class T>
CStoreScalarAggregator<T>::~CStoreScalarAggregator()
	{
	}

template<class T>
void CStoreScalarAggregator<T>::submit_result(CJobResult* result)
	{
		SG_GCDEBUG("Entering\n")

		// check for proper typecast
		CScalarResult<T>* new_result=dynamic_cast<CScalarResult<T>*>(result);
		if (!new_result)
			SG_ERROR("result is not of CScalarResult type!\n");
		// aggregate it with previous
		m_aggregate+=new_result->get_result();

		SG_GCDEBUG("Leaving\n")
	}

template<>
void CStoreScalarAggregator<bool>::submit_result(CJobResult* result)
	{
		SG_NOTIMPLEMENTED
	}

template<>
void CStoreScalarAggregator<char>::submit_result(CJobResult* result)
	{
		SG_NOTIMPLEMENTED
	}

template<class T>
void CStoreScalarAggregator<T>::finalize()
	{
		m_result=(CJobResult*)(new CScalarResult<T>(m_aggregate));
		SG_REF(m_result);
	}

template class CStoreScalarAggregator<bool>;
template class CStoreScalarAggregator<char>;
template class CStoreScalarAggregator<int8_t>;
template class CStoreScalarAggregator<uint8_t>;
template class CStoreScalarAggregator<int16_t>;
template class CStoreScalarAggregator<uint16_t>;
template class CStoreScalarAggregator<int32_t>;
template class CStoreScalarAggregator<uint32_t>;
template class CStoreScalarAggregator<int64_t>;
template class CStoreScalarAggregator<uint64_t>;
template class CStoreScalarAggregator<float32_t>;
template class CStoreScalarAggregator<float64_t>;
template class CStoreScalarAggregator<floatmax_t>;
template class CStoreScalarAggregator<complex128_t>;
}
