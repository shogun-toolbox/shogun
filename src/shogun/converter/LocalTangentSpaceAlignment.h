/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Evan Shelhamer
 */

#ifndef LOCALTANGENTSPACEALIGNMENT_H_
#define LOCALTANGENTSPACEALIGNMENT_H_
#include <shogun/lib/config.h>
#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Features;
class Distance;

/** @brief class LocalTangentSpaceAlignment used to embed
 * data using Local Tangent Space Alignment (LTSA)
 * algorithm as described in:
 *
 * Zhang, Z., & Zha, H. (2002). Principal Manifolds
 * and Nonlinear Dimension Reduction via Local Tangent Space Alignment.
 * Journal of Shanghai University English Edition, 8(4), 406-424. SIAM.
 * Retrieved from http://arxiv.org/abs/cs/0212008
 *
 * This algorithm is pretty stable for variations of k parameter value but
 * be sure it is set with a consistent value (at least 3-5) for reasonable
 * results.
 *
 * Uses implementation from the Tapkee library.
 *
 */
class LocalTangentSpaceAlignment: public LocallyLinearEmbedding
{
public:

	/** constructor */
	LocalTangentSpaceAlignment();

	/** destructor */
	virtual ~LocalTangentSpaceAlignment();

	/** get name */
	virtual const char* get_name() const;

	/** transform */
	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true);
};
}

#endif /* LOCALTANGENTSPACEALINGMENT_H_ */
