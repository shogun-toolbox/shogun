/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef SCALAR_RESULT_H_
#define SCALAR_RESULT_H_

#include <shogun/lib/config.h>
#include <shogun/lib/computation/job/JobResult.h>

namespace shogun
{

/** @brief Base class that stores the result of an independent job
 * when the result is a scalar.
 */
template<class T> class CScalarResult : public CJobResult
{
public:
	/** default constructor, no args */
	CScalarResult()
	: CJobResult(), m_result(static_cast<T>(0))
	{
		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** constructor
	 * @param result the scalar result
	 */
	CScalarResult(const T& result)
	: CJobResult(), m_result(result)
	{
		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** destructor */
	virtual ~CScalarResult()
	{
		SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "CScalarResult";
	}

	/** @return the job result */
	const T get_result() const
	{
		return m_result;
	}

protected:
	/** the scalar result */
	T m_result;
};

template class CScalarResult<bool>;
template class CScalarResult<char>;
template class CScalarResult<int8_t>;
template class CScalarResult<uint8_t>;
template class CScalarResult<int16_t>;
template class CScalarResult<uint16_t>;
template class CScalarResult<int32_t>;
template class CScalarResult<uint32_t>;
template class CScalarResult<int64_t>;
template class CScalarResult<uint64_t>;
template class CScalarResult<float32_t>;
template class CScalarResult<float64_t>;
template class CScalarResult<floatmax_t>;
template class CScalarResult<complex64_t>;

}

#endif // SCALAR_RESULT_H_
