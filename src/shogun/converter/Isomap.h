/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kunal Arora, Sergey Lisitsyn, Heiko Strathmann, Evan Shelhamer
 */

#ifndef ISOMAP_H_
#define ISOMAP_H_
#include <shogun/lib/config.h>
#include <shogun/converter/MultidimensionalScaling.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Features;
class Distance;

/** @brief The Isomap class is used to embed data using Isomap algorithm
 * as described in:
 *
 * Silva, V. D., & Tenenbaum, J. B. (2003).
 * Global versus local methods in nonlinear dimensionality reduction.
 * Advances in Neural Information Processing Systems 15, 15(Figure 2), 721-728. MIT Press.
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.3407&rep=rep1&type=pdf
 *
 * The Isomap algorithm can be considered as a modification of the
 * classic Multidimensional Scaling algorithm. The algorithm itself
 * consists of the following steps:
 * - For each feature vector \f$x\in X\f$ find its \f$k\f$ nearest
 *   neighbours and construct the sparse neighbourhood graph.
 * - Compute the squared distances matrix \f$D\f$ such as \f$D\{i,j\} = d^2(x_i,x_j)\f$.
 * - Relax distances with shortest (so-called geodesic) distances on the sparse neighbourhood graph (e.g. with Dijkstra's algorithm).
 * - Center the matrix \f$D\f$ with subtracting row mean, column mean and adding to the grand mean. Multiply \f$D\f$ element-wise with \f$-0.5\f$.
 * - Compute embedding with the \f$t\f$ eigenvectors that correspond to the largest eigenvalues of the matrix \f$D\f$; normalize these vectors
 *   dividing each eigenvector by the square root of its corresponding eigenvalue. Form the final embedding with eigenvectors as rows and projected
 *   feature vectors as columns.

 * It is possible to apply preprocessor to specified distance using
 * apply_to_distance.
 *
 * Uses implementation from the Tapkee library.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','isomap',k);

 * cf. http://tapkee.lisitsyn.me/
 * cf. https://en.wikipedia.org/wiki/Isomap
 *
 */
class Isomap: public MultidimensionalScaling
{
public:

	/* constructor */
	Isomap();

	/* destructor */
	virtual ~Isomap();

	/** get name */
	const char* get_name() const;

	/** setter for k parameter
	 * @param k value
	 */
	void set_k(int32_t k);

	/** getter for k parameter
	 * @return k value
	 */
	int32_t get_k() const;

	/** embed distance */
	virtual std::shared_ptr<DenseFeatures<float64_t>> embed_distance(std::shared_ptr<Distance> distance);

/// HELPERS
protected:

	/** default init */
	virtual void init();

/// FIELDS
protected:

	/** k, number of neighbors for K-Isomap */
	int32_t m_k;

};
}
#endif /* ISOMAP_H_ */
