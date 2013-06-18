/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef VECTOR_RESULT_H_
#define VECTOR_RESULT_H_

#include <shogun/lib/config.h>
#include <shogun/lib/computation/job/JobResult.h>

namespace shogun
{
template<class T> class SGVector;

/** @brief Base class that stores the result of an independent job
 * when the result is a vector.
 */
template<class T> class CVectorResult : public CJobResult
{
public:
	/** default constructor, no args */
	CVectorResult()
	: CJobResult()
	{
		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** default constructor, one args */
	CVectorResult(SGVector<T> result)
	: CJobResult(), m_result(result)
	{
		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** destructor */
	virtual ~CVectorResult()
	{
		SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "CVectorResult";
	}

	/** @return the job result */
	SGVector<T> get_result() const
	{
		return m_result;
	}

protected:
	/** the vector result */
	SGVector<T> m_result;
};

template class CVectorResult<bool>;
template class CVectorResult<char>;
template class CVectorResult<int8_t>;
template class CVectorResult<uint8_t>;
template class CVectorResult<int16_t>;
template class CVectorResult<uint16_t>;
template class CVectorResult<int32_t>;
template class CVectorResult<uint32_t>;
template class CVectorResult<int64_t>;
template class CVectorResult<uint64_t>;
template class CVectorResult<float32_t>;
template class CVectorResult<float64_t>;
template class CVectorResult<floatmax_t>;
template class CVectorResult<complex64_t>;

}

#endif // VECTOR_RESULT_H_
