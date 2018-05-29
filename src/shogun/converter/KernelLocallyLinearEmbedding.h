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

class CFeatures;
class CKernel;

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
class CKernelLocallyLinearEmbedding: public CLocallyLinearEmbedding
{
public:

	/** constructor */
	CKernelLocallyLinearEmbedding();

	/** constructor
	 * @param kernel kernel to be used
	 */
	CKernelLocallyLinearEmbedding(CKernel* kernel);

	/** destructor */
	virtual ~CKernelLocallyLinearEmbedding();

	/** transform */
	virtual CFeatures* transform(CFeatures* features, bool inplace = true);

	/** embed kernel (kernel should be inited)
	 * @param kernel kernel to construct embed
	 */
	CDenseFeatures<float64_t>* embed_kernel(CKernel* kernel);

	/** get name */
	virtual const char* get_name() const;

};
}

#endif /* KERNELLOCALLYLINEAREMBEDDING_H_ */
