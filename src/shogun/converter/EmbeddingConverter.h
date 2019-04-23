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

class Features;
class Distance;
class Kernel;

/** @brief class EmbeddingConverter (part of the Efficient Dimensionality
 * Reduction Toolkit) used to construct embeddings of
 * features, e.g. construct dense numeric embedding of string features
 */
class EmbeddingConverter: public Converter
{
public:

	/** constructor */
	EmbeddingConverter();

	/** destructor */
	virtual ~EmbeddingConverter();

	/** Apply transformation to features. In-place mode is not supported for
	 * Tapkee converters.
	 * @param features features to embed
	 * @return embedding dense real features
	 */
	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true) = 0;

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
	void set_distance(std::shared_ptr<Distance> distance);

	/** getter for distance
	 * @return distance
	 */
	std::shared_ptr<Distance> get_distance() const;

	/** setter for kernel
	 * @param kernel kernel to set
	 */
	void set_kernel(std::shared_ptr<Kernel> kernel);

	/** getter for kernel
	 * @return kernel
	 */
	std::shared_ptr<Kernel> get_kernel() const;

	virtual const char* get_name() const { return "EmbeddingConverter"; };

protected:

	/** default init */
	void init();

protected:

	/** target dim of dimensionality reduction preprocessor */
	int32_t m_target_dim;

	/** distance to be used */
	std::shared_ptr<Distance> m_distance;

	/** kernel to be used */
	std::shared_ptr<Kernel> m_kernel;
};
}

#endif /* DIMENSIONREDUCTIONPREPROCESSOR_H_ */
