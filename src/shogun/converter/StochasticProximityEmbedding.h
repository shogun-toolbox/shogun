/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef STOCHASTICPROXIMITYEMBEDDING_H_
#define STOCHASTICPROXIMITYEMBEDDING_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

/** Stochastic Proximity Embedding (SPE) strategy */
enum ESPEStrategy
{
	SPE_GLOBAL,
	SPE_LOCAL,
};

/** @brief class StochasticProximityEmbedding (part of the Efficient 
 * Dimensionality Reduction Toolkit) used to construct embeddings of data using 
 * the Stochastic Proximity algorithm.
 *
 * TODO more documentation
 *
 */

class CStochasticProximityEmbedding : public CEmbeddingConverter
{

	public:

		/** constructor */
		CStochasticProximityEmbedding();

		/** destructor */
		virtual ~CStochasticProximityEmbedding();

		/** apply to features
		 *
		 * @param features features to embed
		 * @return embedding features
		 */
		virtual CFeatures* apply(CFeatures* features);

		/** setter for K parameter
		 *
		 * @param k k value
		 */
		inline void set_k(int32_t k);

		/** getter for K parameter
		 *
		 * @return k value
		 */
		inline int32_t get_k() const;

		/** setter for regularization parameter
		 *
		 * @param m_tolerance regularization value
		 */
		inline void set_tolerance(float32_t tolerance);

		/** getter for regularization parameter
		 *
		 * @return regularization value
		 */
		inline int32_t get_tolerance() const;

		/** get name */
		inline virtual const char* get_name() const;

	private:

		/** default init */
		void init();

	private:

		/** SPE strategy */
		ESPEStrategy m_strategy;

		/** number of neighbours in local strategy */
		int32_t m_k;

		/** regularization parameter */
		float32_t m_tolerance;

};

} /* namespace shogun */


#endif /* HAVE_LAPACK */
#endif /* STOCHASTICPROXIMITYEMBEDDING_H_ */ 
