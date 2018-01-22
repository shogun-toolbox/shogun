/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sunil Mahendrakar, Soumyajit De, Björn Esser
 */

#ifndef OPERATOR_FUNCTION_H_
#define OPERATOR_FUNCTION_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>

namespace shogun
{

/** linear operator function types */
enum EOperatorFunction
{
	OF_SQRT=0,
	OF_LOG=1,
	OF_POLY=2,
	OF_UNDEFINED=3
};

template<class T> class SGVector;
template<class T> class CLinearOperator;

/** @brief Abstract template base class for computing \f$s^{T} f(C) s\f$ for a
 * linear operator C and a vector s.
 */
template<class T> class COperatorFunction : public CSGObject
{
public:
	/** default constructor */
	COperatorFunction()
	: CSGObject(),
	  m_function_type(OF_UNDEFINED)
	{
		init();
	}

	/**
	 * constructor
	 *
	 * @param op the linear operator of this operator function
	 * @param type the type of the operator function (sqrt, log, etc)
	 */
	COperatorFunction(CLinearOperator<T>* op,
		EOperatorFunction type=OF_UNDEFINED)
	: CSGObject(),
	  m_function_type(type)
	{
		init();

		m_linear_operator=op;
		SG_REF(m_linear_operator);
	}

	/** destructor */
	virtual ~COperatorFunction()
	{
		SG_UNREF(m_linear_operator);
	}

	/** @return the operator */
	CLinearOperator<T>* get_operator() const
	{
		return m_linear_operator;
	}

	/**
	 * abstract precompute method that must be called before using solve
	 * for performing preliminary computations that are necessary.
	 */
	virtual void precompute() = 0;

	/**
	 * method that solves for a sample and returns the final result
	 */
	virtual float64_t solve(SGVector<T> sample) = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "OperatorFunction";
	}
protected:
	/** the linear operator */
	CLinearOperator<T>* m_linear_operator;

	/** the linear operator function type */
	const EOperatorFunction m_function_type;

private:
	/** initialize with default values and register params */
	void init()
	{
	  m_linear_operator=NULL;

		SG_ADD((CSGObject**)&m_linear_operator, "linear_operator",
			"Linear operator of this operator function", MS_NOT_AVAILABLE);
	}
};
}

#endif // OPERATOR_FUCNTION_H_
