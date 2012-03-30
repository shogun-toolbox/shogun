/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Sergey Lisitsyn
 */

#ifndef LINEARLOCALTANGENTSPACEALIGNMENT_H_
#define LINEARLOCALTANGENTSPACEALIGNMENT_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/LocalTangentSpaceAlignment.h>
#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class LinearLocalTangentSpaceAlignment (part of the
 * Efficient Dimensionality Reduction Toolkit) converter used to
 * construct embeddings as described in:
 *
 * Zhang, T., Yang, J., Zhao, D., & Ge, X. (2007).
 * Linear local tangent space alignment and application to face recognition.
 * Neurocomputing, 70(7-9), 1547-1553.
 * Retrieved from http://linkinghub.elsevier.com/retrieve/pii/S0925231206004577
 *
 * This method is hardly applicable to very high-dimensional data due to
 * necessity to solve eigenproblem involving matrix of size (dim x dim).
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','lltsa',k);
 *
 */
class CLinearLocalTangentSpaceAlignment: public CLocalTangentSpaceAlignment
{
public:

	/** constructor */
	CLinearLocalTangentSpaceAlignment();

	/** destructor */
	virtual ~CLinearLocalTangentSpaceAlignment();

	/** get name */
	virtual const char* get_name() const;

protected:

	/** constructs embedding
	 * @param simple features to be used
	 * @param matrix weight matrix
	 * @param dimension dimension of embedding
	 * @return null-space approximation feature matrix
	 */
	virtual SGMatrix<float64_t> construct_embedding(CFeatures* features, SGMatrix<float64_t> matrix, int dimension);
};
}

#endif /* HAVE_LAPACK */
#endif /* LINEARLOCALTANGENTSPACEALIGNMENT_H_ */
