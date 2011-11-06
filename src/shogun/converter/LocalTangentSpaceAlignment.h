/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef LOCALTANGENTSPACEALIGNMENT_H_
#define LOCALTANGENTSPACEALIGNMENT_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief the class LocalTangentSpaceAlignment used to preprocess
 * data using Local Tangent Space Alignment (LTSA) algorithm as described in:
 *
 * Zhang, Z., & Zha, H. (2002). Principal Manifolds 
 * and Nonlinear Dimension Reduction via Local Tangent Space Alignment. 
 * Journal of Shanghai University English Edition, 8(4), 406-424. SIAM. 
 * Retrieved from http://arxiv.org/abs/cs/0212008
 *
 * The stated eigenproblem is solved in the same way as
 * CLocallyLinearEmbedding (LAPACK or ARPACK if available).
 *
 * The local tangent space alignment step is parallel. Neighborhood
 * determination is parallel as in CLocallyLinearEmbedding.
 *
 * This algorithm is pretty stable for variations of k parameter value but
 * be sure it is set with a consistent value (at least 3-5) for reasonable
 * results.
 *
 * Please do not use multithreading whether your LAPACK is not thread-safe.
 *
 */
class CLocalTangentSpaceAlignment: public CLocallyLinearEmbedding
{
public:

	/** constructor */
	CLocalTangentSpaceAlignment();

	/** destructor */
	virtual ~CLocalTangentSpaceAlignment();

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

	/** run ltsa thread
	 * @param p thread params
	 */
	static void* run_ltsa_thread(void* p);

};
}

#endif /* HAVE_LAPACK */
#endif /* LOCALTANGENTSPACEALINGMENT_H_ */
