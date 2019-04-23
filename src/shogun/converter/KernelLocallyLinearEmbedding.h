/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Evan Shelhamer
 */

#ifndef KERNELLOCALLYLINEAREMBEDDING_H_
#define KERNELLOCALLYLINEAREMBEDDING_H_
#include <shogun/lib/config.h>
#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

class Features;
class Kernel;

/** @brief class KernelLocallyLinearEmbedding used to construct embeddings
 * of data using kernel formulation of Locally Linear Embedding algorithm as
 * described in
 *
 * Decoste, D. (2001).
 * Visualizing Mercer Kernel Feature Spaces Via Kernelized Locally-Linear Embeddings.
 * The 8th International Conference on Neural Information Processing ICONIP2001
 *
 * It is optimized with the alignment formulation as described in
 *
 * Zhao, D. (2006).
 * Formulating LLE using alignment technique.
 * Pattern Recognition, 39(11), 2233-2235.
 * Retrieved from http://linkinghub.elsevier.com/retrieve/pii/S0031320306002160
 *
 * Uses the Tapkee library code.
 *
 */
class KernelLocallyLinearEmbedding: public LocallyLinearEmbedding
{
public:

	/** constructor */
	KernelLocallyLinearEmbedding();

	/** constructor
	 * @param kernel kernel to be used
	 */
	KernelLocallyLinearEmbedding(std::shared_ptr<Kernel> kernel);

	/** destructor */
	virtual ~KernelLocallyLinearEmbedding();

	/** transform */
	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true);

	/** embed kernel (kernel should be inited)
	 * @param kernel kernel to construct embed
	 */
	std::shared_ptr<DenseFeatures<float64_t>> embed_kernel(std::shared_ptr<Kernel> kernel);

	/** get name */
	virtual const char* get_name() const;

};
}

#endif /* KERNELLOCALLYLINEAREMBEDDING_H_ */
