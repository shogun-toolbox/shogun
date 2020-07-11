/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Evan Shelhamer, Chiyuan Zhang
 */

#ifndef LOCALLYLINEAREMBEDDING_H_
#define LOCALLYLINEAREMBEDDING_H_
#include <shogun/lib/config.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Features;
class Distance;

/** @brief class LocallyLinearEmbedding used to embed
 * data using Locally Linear Embedding algorithm described in
 *
 * Saul, L. K., Ave, P., Park, F., & Roweis, S. T. (2001).
 * An Introduction to Locally Linear Embedding. Available from, 290(5500), 2323-2326.
 * Retrieved from:
 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.123.7319&rep=rep1&type=pdf
 *
 * It is optimized with alignment formulation as described in
 *
 * Zhao, D. (2006).
 * Formulating LLE using alignment technique.
 * Pattern Recognition, 39(11), 2233-2235.
 * Retrieved from http://linkinghub.elsevier.com/retrieve/pii/S0031320306002160
 *
 * Uses implementation from the Tapkee library.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','lle',k);
 *
 */
class LocallyLinearEmbedding: public EmbeddingConverter
{
public:

	/** constructor */
	LocallyLinearEmbedding();

	/** destructor */
	~LocallyLinearEmbedding() override;

	/** apply preprocessor to features
	 * @param features
	 */
	std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true) override;

	/** setter for k parameter
	 * @param k k value
	 */
	void set_k(int32_t k);

	/** getter for k parameter
	 * @return m_k value
	 */
	int32_t get_k() const;

	/** setter for reconstruction shift parameter
	 * @param reconstruction_shift reconstruction shift value
	 */
	void set_reconstruction_shift(float64_t reconstruction_shift);

	/** getter for reconstruction shift parameter
	 * @return m_reconstruction_shift value
	 */
	float64_t get_reconstruction_shift() const;

	/** setter for nullspace shift parameter
	 * @param nullspace_shift nullsapce shift value
	 */
	void set_nullspace_shift(float64_t nullspace_shift);

	/** getter for nullspace shift parameter
	 * @return m_nullspace_shift value
	 */
	float64_t get_nullspace_shift() const;

	/** get name */
	const char* get_name() const override;

	/// HELPERS
protected:

	/** default init */
	void init();

	/// FIELDS
protected:

	/** number of neighbors */
	int32_t m_k;

	/** regularization shift of reconstruction step */
	float64_t m_reconstruction_shift;

	/** regularization shift of nullspace finding step */
	float64_t m_nullspace_shift;

};
}

#endif /* LOCALLYLINEAREMBEDDING_H_ */
