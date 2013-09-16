/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef NORMAL_SAMPLER_H_
#define NORMAL_SAMPLER_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/TraceSampler.h>

namespace shogun
{
template<class T> class SGVector;

/** @brief Class that provides a sample method for Gaussian samples 
 */
class CNormalSampler : public CTraceSampler
{
public:
	/** default constructor */
	CNormalSampler();

	/** constructor 
	 * @param dimension the dimension of the Gaussian sample vectors ~(0,I)
	 */
	CNormalSampler(index_t dimension);

	/** destructor */
	virtual ~CNormalSampler();

	/** method that generates the samples
 	 * @param idx the index (this is effectively ignored)
	 * @return the sample vector
 	 */
	virtual SGVector<float64_t> sample(index_t idx) const;

	/** precompute method that sets the num_samples of the base */
	virtual void precompute();

	/** @return object name */
	virtual const char* get_name() const
	{
		return "NormalSampler";
	}
};

}

#endif // NORMAL_SAMPLER_H_
