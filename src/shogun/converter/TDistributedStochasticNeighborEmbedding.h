/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Vladyslav S. Gorbatiuk
 * Copyright (C) 2011-2013 Vladyslav S. Gorbatiuk
 */

#ifndef TDISTRIBUTEDSTOCHASTICNEIGHBOREMBEDDING_H_
#define TDISTRIBUTEDSTOCHASTICNEIGHBOREMBEDDING_H_
#include <shogun/lib/config.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>

namespace shogun
{

/** @brief class CTDistributedStochasticNeighborEmbedding used to embed
 * data using t-distributed stochastic neighbor embedding algorithm:
 * http://jmlr.csail.mit.edu/papers/volume9/vandermaaten08a/vandermaaten08a.pdf.
 *
 * Uses implementation from the Tapkee library.
 *
 */
class CTDistributedStochasticNeighborEmbedding : public CEmbeddingConverter
{
public:

	/** constructor */
	CTDistributedStochasticNeighborEmbedding();

	/** destructor */
	virtual ~CTDistributedStochasticNeighborEmbedding();

	/** get name */
	virtual const char* get_name() const;

	/** apply preprocessor to features
	 *
	 * @param features features to embed
	 */
	virtual CFeatures* apply(CFeatures* features);

	/** setter for the learning rate
	 *
	 * @param theta the learning rate
	 */
	void set_theta(const float64_t theta);

	/** getter for the learning rate
	 *
	 * @return the learning rate theta
	 */
	float64_t get_theta() const;

	/** setter for perplexity
	 *
	 * @param perplexity convergence parameter
	 */
	void set_perplexity(const float64_t perplexity);

	/** getter for perplexity
	 *
	 * @return perplexity
	 */
	float64_t get_perplexity() const;

private:

	/** default init */
	void init();

private:

	/** theta - learning rate */
	float64_t m_theta;

	/** perplexity */
	float64_t m_perplexity;

}; /* class CTDistributedStochasticNeighborEmbedding */

} /* namespace shogun */

#endif /* TDISTRIBUTEDSTOCHASTICNEIGHBOREMBEDDING_H_ */
