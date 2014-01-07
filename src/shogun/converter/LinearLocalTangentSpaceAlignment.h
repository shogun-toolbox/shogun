/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Sergey Lisitsyn
 */

#ifndef LINEARLOCALTANGENTSPACEALIGNMENT_H_
#define LINEARLOCALTANGENTSPACEALIGNMENT_H_
#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <converter/LocalTangentSpaceAlignment.h>
#include <features/Features.h>
#include <features/DenseFeatures.h>
#include <distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;

/** @brief class LinearLocalTangentSpaceAlignment converter used to
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
 * Uses implementation from the Tapkee library.
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

	/** apply */
	virtual CFeatures* apply(CFeatures* features);

};
}

#endif /* HAVE_EIGEN3 */
#endif /* LINEARLOCALTANGENTSPACEALIGNMENT_H_ */
