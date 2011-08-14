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
#ifdef HAVE_LAPACK
#include <shogun/preprocessor/LocallyLinearEmbedding.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;

class CDistance;

/** @brief the class LocalTangentSpaceAlignment used to preprocess
 * data using Local Tangent Space Alignment (LTSA) algorithm described in
 *
 * Z. Zhang, H. Zha, Principal manifolds and nonlinear dimensionality
 * reduction via tangent space alignment, SIAM J. Sci. Comput. 26 (1)
 * (2004) 313–338.
 *
 */
class CLocalTangentSpaceAlignment: public CLocallyLinearEmbedding
{
public:

	/** constructor */
	CLocalTangentSpaceAlignment();

	/** destructor */
	virtual ~CLocalTangentSpaceAlignment();

	/** init
	 * @param data feature vectors for preproc
	 */
	virtual bool init(CFeatures* features);

	/** cleanup
	 *
	 */
	virtual void cleanup();

	/** apply preproc to feature matrix
	 *
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

	/** apply preproc to feature vector
	 *
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

	/** get name */
	virtual inline const char* get_name() const { return "LocalTangentSpaceAlignment"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_LOCALTANGENTSPACEALIGNMENT; };

protected:

	/** run ltsa thread
	 * @param p thread params
	 */
	static void* run_ltsa_thread(void* p);

};
}

#endif /* HAVE_LAPACK */
#endif /* LOCALTANGENTSPACEALINGMENT_H_ */
