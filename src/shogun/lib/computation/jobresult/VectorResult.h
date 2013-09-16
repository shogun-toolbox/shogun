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
#include <shogun/lib/SGVector.h>
#include <shogun/lib/computation/jobresult/JobResult.h>
#include <shogun/base/Parameter.h>

namespace shogun
{
/** @brief Base class that stores the result of an independent job
 * when the result is a vector.
 */
template<class T> class CVectorResult : public CJobResult
{
/** this class supports complex */
typedef bool supports_complex64_t;

public:
	/** default constructor */
	CVectorResult()
	: CJobResult()
	{
		init();
	}

	/** constructor
	 * @param result the vector result
	 */
	CVectorResult(SGVector<T> result)
	: CJobResult(), m_result(result)
	{
		init();
	}

	/** destructor */
	virtual ~CVectorResult()
	{
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "VectorResult";
	}

	/** @return the job result */
	SGVector<T> get_result() const
	{
		return m_result;
	}

protected:
	/** the vector result */
	SGVector<T> m_result;

private:
	/** initialize with default values and register params */
	void init()
	{
		set_generic<T>();

		SG_ADD(&m_result, "result",
			"The result vector", MS_NOT_AVAILABLE);
	}
};
}

#endif // VECTOR_RESULT_H_
