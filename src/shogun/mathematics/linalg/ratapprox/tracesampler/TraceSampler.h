/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */
#ifndef TRACE_SAMPLER_H_
#define TRACE_SAMPLER_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>

namespace shogun
{
template<class T> class SGVector;

/** @brief Abstract template base class that provides an interface for sampling
 * the trace of a linear operator using an abstract sample method
 */
class CTraceSampler : public CSGObject
{
public:
	/** default constructor */
	CTraceSampler()
	: CSGObject()
	{
		init();

		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** 
	 * constructor
	 *
	 * @param dimension the dimension of the sample vectors
	 */
	CTraceSampler(index_t dimension)
	: CSGObject()
	{
		init();

		m_dimension=dimension;

		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** destructor */
	virtual ~CTraceSampler()
	{
		SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
	}

	/** 
	 * abstract method that generates the samples
	 *
 	 * @param idx the index which determines which sample to draw
	 * @return the sample vector
 	 */
	virtual SGVector<float64_t> sample(index_t idx) const = 0;

	/**
	 * abstract method for initializing the sampler, number of samples etc,
	 * must be called before sample
	 */
	virtual void precompute() = 0;

	/** @return the number of samples */
	virtual const index_t get_num_samples() const
	{
		return m_num_samples;
	}

	/** @return the number of samples */
	virtual const index_t get_dimension() const
	{
		return m_dimension;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "TraceSampler";
	}

protected:
	/** the dimension of the sample vectors */
	index_t m_dimension;

	/** the number of samples this sampler will generate, set by implementation */
	index_t m_num_samples;

private:
	/** initialize with default values and register params */
	void init()
	{
		m_num_samples=0;
		m_dimension=0;

		SG_ADD(&m_num_samples, "num_samples",
			"Number of samples this sampler can generate", MS_NOT_AVAILABLE);

		SG_ADD(&m_dimension, "sample_dimension",
			"Dimension of samples this sampler can generate", MS_NOT_AVAILABLE);
	}
};

}

#endif // TRACE_SAMPLER_H_
