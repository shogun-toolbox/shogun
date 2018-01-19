/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Thoralf Klein, Soeren Sonnenburg, Yuyu Zhang, 
 *          Björn Esser
 */

#ifndef SCALAR_RESULT_H_
#define SCALAR_RESULT_H_

#include <shogun/lib/config.h>

#include <shogun/lib/computation/jobresult/JobResult.h>
#include <shogun/io/SGIO.h>

namespace shogun
{

/** @brief Base class that stores the result of an independent job
 * when the result is a scalar.
 */
template<class T> class CScalarResult : public CJobResult
{
/** this class supports complex */
typedef bool supports_complex128_t;

public:
	/** default constructor */
	CScalarResult()
	: CJobResult()
	{
		init();

		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/**
	 * constructor
	 *
	 * @param scalar_result the scalar result
	 */
	CScalarResult(const T& scalar_result)
	: CJobResult()
	{
		init();

		m_result=scalar_result;

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
		return "ScalarResult";
	}

	/** @return the job result */
	const T get_result() const
	{
		return m_result;
	}

protected:
	/** the scalar result */
	T m_result;

private:
	/** initialize with default values and register params */
	void init()
	{
		m_result=static_cast<T>(0);

		set_generic<T>();

		SG_ADD(&m_result, "scalar_result", "Scalar result of a computation job",
			MS_NOT_AVAILABLE);
	}
};
}

#endif // SCALAR_RESULT_H_
