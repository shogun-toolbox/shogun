/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 */

#ifndef KERNELLOCALTANGENTSPACEALIGNMENT_H_
#define KERNELLOCALTANGENTSPACEALIGNMENT_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/preprocessor/KernelLocallyLinearEmbedding.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief the class LocalTangentSpaceAlignment used to preprocess
 * data using kernel extension of the Local Tangent Space Alignment (LTSA) algorithm.
 *
 * The stated eigenproblem is solved in the same way as
 * CLocallyLinearEmbedding (LAPACK or ARPACK if available).
 *
 * The local tangent space alignment step is parallel. Neighborhood
 * determination is parallel as in CLocallyLinearEmbedding.
 */
class CKernelLocalTangentSpaceAlignment: public CKernelLocallyLinearEmbedding
{
public:

	/** constructor */
	CKernelLocalTangentSpaceAlignment();

	/** destructor */
	virtual ~CKernelLocalTangentSpaceAlignment();

	/** get name */
	virtual const char* get_name() const;

	/** get type */
	virtual EPreprocessorType get_type() const;

/// HELPERS
protected:

	/** construct weight matrix 
	 */
	virtual SGMatrix<float64_t> construct_weight_matrix(SGMatrix<float64_t> kernel_matrix, 
	                                                    SGMatrix<int32_t> neighborhood_matrix);

/// THREADS
protected:

	/** run kernel ltsa thread
	 * @param p thread params
	 */
	static void* run_kltsa_thread(void* p);

};
}

#endif /* HAVE_LAPACK */
#endif /* KERNELLOCALTANGENTSPACEALINGMENT_H_ */
