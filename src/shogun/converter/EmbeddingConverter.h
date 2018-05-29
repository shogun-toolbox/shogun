/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Evan Shelhamer, Yuyu Zhang
 */

#ifndef EMBEDDINGCONVERTER_H_
#define EMBEDDINGCONVERTER_H_

#include <shogun/lib/config.h>

#include <shogun/converter/Converter.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/Distance.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

class CFeatures;
class CDistance;
class CKernel;

/** @brief class EmbeddingConverter (part of the Efficient Dimensionality
 * Reduction Toolkit) used to construct embeddings of
 * features, e.g. construct dense numeric embedding of string features
 */
class CEmbeddingConverter: public CConverter
{
public:

	/** constructor */
	CEmbeddingConverter();

	/** destructor */
	virtual ~CEmbeddingConverter();

	/** Apply transformation to features. In-place mode is not supported for
	 * Tapkee converters.
	 * @param features features to embed
	 * @return embedding features
	 */
	virtual CFeatures* transform(CFeatures* features, bool inplace = true) = 0;

	/** embed given features, acts the same as apply, but returns
	 * DenseFeatures
	 * @param features features to embed
	 * @return embedding simple features
	 */
	virtual CDenseFeatures<float64_t>* embed(CFeatures* features);

	/** setter for target dimension
	 * @param dim target dimension
	 */
	void set_target_dim(int32_t dim);

	/** getter for target dimension
	 * @return target dimension
	 */
	int32_t get_target_dim() const;

	/** setter for distance
	 * @param distance distance to set
	 */
	void set_distance(CDistance* distance);

	/** getter for distance
	 * @return distance
	 */
	CDistance* get_distance() const;

	/** setter for kernel
	 * @param kernel kernel to set
	 */
	void set_kernel(CKernel* kernel);

	/** getter for kernel
	 * @return kernel
	 */
	CKernel* get_kernel() const;

	virtual const char* get_name() const { return "EmbeddingConverter"; };

protected:

	/** default init */
	void init();

protected:

	/** target dim of dimensionality reduction preprocessor */
	int32_t m_target_dim;

	/** distance to be used */
	CDistance* m_distance;

	/** kernel to be used */
	CKernel* m_kernel;
};
}

#endif /* DIMENSIONREDUCTIONPREPROCESSOR_H_ */
