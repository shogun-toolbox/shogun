/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Evan Shelhamer, Bjoern Esser
 */

#ifndef LINEARLOCALTANGENTSPACEALIGNMENT_H_
#define LINEARLOCALTANGENTSPACEALIGNMENT_H_
#include <shogun/lib/config.h>
#include <shogun/converter/LocalTangentSpaceAlignment.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Features;
class Distance;

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
class LinearLocalTangentSpaceAlignment: public LocalTangentSpaceAlignment
{
public:

	/** constructor */
	LinearLocalTangentSpaceAlignment();

	/** destructor */
	virtual ~LinearLocalTangentSpaceAlignment();

	/** get name */
	virtual const char* get_name() const;

	/** transform */
	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true);
};
}

#endif /* LINEARLOCALTANGENTSPACEALIGNMENT_H_ */
