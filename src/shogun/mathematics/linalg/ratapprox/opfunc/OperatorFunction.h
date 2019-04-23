/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sunil Mahendrakar, Soumyajit De, Bjoern Esser
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
template<class T> class LinearOperator;

/** @brief Abstract template base class for computing \f$s^{T} f(C) s\f$ for a
 * linear operator C and a vector s.
 */
template<class T> class OperatorFunction : public SGObject
{
public:
	/** default constructor */
	OperatorFunction()
	: SGObject(),
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
	OperatorFunction(std::shared_ptr<LinearOperator<T>> op,
		EOperatorFunction type=OF_UNDEFINED)
	: SGObject(),
	  m_function_type(type)
	{
		init();

		m_linear_operator=op;
	}

	/** destructor */
	virtual ~OperatorFunction()
	{
	}

	/** @return the operator */
	std::shared_ptr<LinearOperator<T>> get_operator() const
	{
		return m_linear_operator;
	}

	/**
	 * purely virtual method that must be called before compute
	 * for performing preliminary computations that are necessary.
	 */
	virtual void precompute() = 0;

	/**
	 * Method that computes for a sample and returns the final result
	 */
	virtual float64_t compute(SGVector<T> sample) const = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "OperatorFunction";
	}
protected:
	/** the linear operator */
	std::shared_ptr<LinearOperator<T>> m_linear_operator;

	/** the linear operator function type */
	const EOperatorFunction m_function_type;

private:
	/** initialize with default values and register params */
	void init()
	{
	  m_linear_operator=NULL;

		SG_ADD((std::shared_ptr<SGObject>*)&m_linear_operator, "linear_operator",
			"Linear operator of this operator function");
	}
};
}

#endif // OPERATOR_FUCNTION_H_
