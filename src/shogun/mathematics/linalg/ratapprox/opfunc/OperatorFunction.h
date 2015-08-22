/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef OPERATOR_FUNCTION_H_
#define OPERATOR_FUNCTION_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/lib/computation/engine/IndependentComputationEngine.h>

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
class CJobResultAggregator;
template<class T> class CLinearOperator;

/** @brief Abstract template base class for computing \f$s^{T} f(C) s\f$ for a
 * linear operator C and a vector s. submit_jobs method creates a bunch of jobs
 * needed to solve for this particular \f$s\f$ and attaches one unique job
 * aggregator to each of them, then submits them all to the computation engine.
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
	 * @param engine the computation engine for the independent jobs
	 * @param type the type of the operator function (sqrt, log, etc)
	 */
	COperatorFunction(CLinearOperator<T>* op,
		CIndependentComputationEngine* engine,
		EOperatorFunction type=OF_UNDEFINED)
	: CSGObject(),
	  m_function_type(type)
	{
		init();

		m_linear_operator=op;
		SG_REF(m_linear_operator);

		m_computation_engine=engine;
		SG_REF(m_computation_engine);
	}

	/** destructor */
	virtual ~COperatorFunction()
	{
		SG_UNREF(m_linear_operator);
		SG_UNREF(m_computation_engine);
	}

	/** @return the operator */
	CLinearOperator<T>* get_operator() const
	{
		return m_linear_operator;
	}

	/**
	 * abstract precompute method that must be called before using submit jobs
	 * for performing preliminary computations that are necessary for the
	 * rest of the computation jobs
	 */
	virtual void precompute() = 0;

	/**
	 * abstract method that creates a job result aggregator, then creates a
	 * number of jobs based on its implementation, attaches the aggregator
	 * with all those jobs, hands over the responsility of those to the
	 * computation engine and then returns the aggregator for collecting the
	 * job results
	 *
	 * @param sample the vector for which new computation job(s) are to be created
	 * @return the array of generated independent jobs
	 */
	virtual CJobResultAggregator* submit_jobs(SGVector<T> sample) = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "OperatorFunction";
	}
protected:
	/** the linear operator */
	CLinearOperator<T>* m_linear_operator;

	/** the computation engine */
	CIndependentComputationEngine* m_computation_engine;

	/** the linear operator function type */
	const EOperatorFunction m_function_type;

private:
	/** initialize with default values and register params */
	void init()
	{
	  m_linear_operator=NULL;
	  m_computation_engine=NULL;

		SG_ADD((CSGObject**)&m_linear_operator, "linear_operator",
			"Linear operator of this operator function", MS_NOT_AVAILABLE);

		SG_ADD((CSGObject**)&m_computation_engine, "computation_engine",
			"Computation engine for the jobs this will generate", MS_NOT_AVAILABLE);
	}
};
}

#endif // OPERATOR_FUCNTION_H_
