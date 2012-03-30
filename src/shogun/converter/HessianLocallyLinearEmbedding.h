/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef HESSIANLOCALLYLINEAREMBEDDING_H_
#define HESSIANLOCALLYLINEAREMBEDDING_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class HessianLocallyLinearEmbedding (part of the
 * Efficient Dimensionality Reduction Toolkit) used to preprocess
 * data using Hessian Locally Linear Embedding algorithm as described in
 *
 * Donoho, D., & Grimes, C. (2003).
 * Hessian eigenmaps: new tools for nonlinear dimensionality reduction.
 * Proceedings of National Academy of Science (Vol. 100, pp. 5591-5596).
 *
 * Stated eigenproblem is solved in the same way as in
 * CLocallyLinearEmbedding (LAPACK or ARPACK if available).
 *
 * The hessian estimation step is parallel and neighborhood determination
 * too as in CLocallyLinearEmbedding.
 *
 * Be sure k value is set with at least
 * 1+[target dim]+1/2 [target_dim]*[1 + target dim], e.g.
 * greater than 6 for target dimensionality of 2.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','hlle',k);
 *
 */
class CHessianLocallyLinearEmbedding: public CLocallyLinearEmbedding
{
public:

	/** constructor */
	CHessianLocallyLinearEmbedding();

	/** destructor */
	virtual ~CHessianLocallyLinearEmbedding();

	/** get name */
	virtual const char* get_name() const;

/// HELPER
protected:

	/** construct weight matrix
	 * @param simple_features features to be used
	 * @param W_matrix weight matrix
	 * @param neighborhood_matrix matrix containing neighbor idxs
	 */
	virtual SGMatrix<float64_t> construct_weight_matrix(CSimpleFeatures<float64_t>* simple_features,float64_t* W_matrix,
                                                            SGMatrix<int32_t> neighborhood_matrix);

/// THREAD
protected:

	/** run hessian estimation thread
	 * @param p thread params
	 */
	static void* run_hessianestimation_thread(void* p);

};
}

#endif /* HAVE_LAPACK */
#endif /* HESSIANLOCALLYLINEAREMBEDDING_H_ */
