/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Heiko Strathmann, Bjoern Esser
 */

#ifndef PROBING_SAMPLER_H_
#define PROBING_SAMPLER_H_

#include <shogun/lib/config.h>

#ifdef HAVE_COLPACK

#include <shogun/mathematics/linalg/ratapprox/tracesampler/TraceSampler.h>
#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{

/** coloring order (see ColPack's GraphColoringInterface for details) */
enum EOrderingVariant
{
	NATURAL=0,
	LARGEST_FIRST,
	DYNAMIC_LARGEST_FIRST,
	DISTANCE_TWO_LARGEST_FIRST,
	SMALLEST_LAST,
	DISTANCE_TWO_SMALLEST_LAST,
	INCIDENCE_DEGREE,
	DISTANCE_TWO_INCIDENCE_DEGREE,
	RANDOM
};

/** coloring variant (see ColPack's GraphColoringInterface for details) */
enum EColoringVariant
{
	DISTANCE_ONE=0,
	ACYCLIC,
	ACYCLIC_FOR_INDIRECT_RECOVERY,
	STAR,
	RESTRICTED_STAR,
	DISTANCE_TWO
};

template<class T> class SGVector;
template<class T> class SparseMatrixOperator;

/** @brief Class that provides a sample method for probing vector using
 * greedy graph coloring. It depends on an external library ColPack (used
 * under GPL2+) for graph coloring related things.
 */
class ProbingSampler : public RandomMixin<TraceSampler>
{
public:
	/** default constructor */
	ProbingSampler();

	/**
	 * constructor
	 *
	 * @param matrix_operator the sparse matrix operator
	 * @param power the power of the sparse matrix operator whose
	 * sparsity structure will be used for grpah coloring
	 * @param ordering the ordering variant
	 * @param coloring the coloring variant
	 */
	ProbingSampler(const std::shared_ptr<SparseMatrixOperator<float64_t>>& matrix_operator,
		int64_t power=1, EOrderingVariant ordering=NATURAL,
		EColoringVariant coloring=DISTANCE_TWO);

	/** destructor */
	virtual ~ProbingSampler();

	/**
	 * set the coloring vector
	 * @param coloring_vector the coloring vector
	 */
	void set_coloring_vector(SGVector<int32_t> coloring_vector);

	/** @return the coloring vector */
	SGVector<int32_t> get_coloring_vector() const;

	/**
	 * method that generates the samples
	 *
	 * @param idx the index
	 * @return the sample vector
	 */
	virtual SGVector<float64_t> sample(index_t idx) const;

	/** precompute method that sets the num_samples of the base */
	virtual void precompute();

	/** @return object name */
	virtual const char* get_name() const
	{
		return "ProbingSampler";
	}

private:
	/** the matrix operator */
	std::shared_ptr<SparseMatrixOperator<float64_t>> m_matrix_operator;

	/** power of the matrix */
	int64_t m_power;

	/** coloring vector */
	SGVector<int32_t> m_coloring_vector;

	/** ordering variant */
	EOrderingVariant m_ordering;

	/** coloring variant */
	EColoringVariant m_coloring;

	/** flag to avoid repeated precompute */
	bool m_is_precomputed;

	/** initialize with default values and register params */
	void init();

};

}

#endif // HAVE_COLPACK
#endif // PROBING_SAMPLER_H_
