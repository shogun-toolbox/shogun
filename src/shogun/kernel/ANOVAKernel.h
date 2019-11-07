/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Heiko Strathmann,
 *          Evan Shelhamer, Bjoern Esser
 */

#include <shogun/lib/config.h>

#ifndef ANOVAKERNEL_H_
#define ANOVAKERNEL_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

class Distance;

/** @brief ANOVA (ANalysis Of VAriances) kernel
 *
 * Formally described as
 *
 * \f[
 *		K_d(x,z) = \sum_{1\le i_1<i_2<\dots<i_d\le n} \prod_{j=1}^d x_{i_j} z_{i_j}
 * \f]
 * with d(cardinality)=1 by default
 * this function is computed recusively
 */

class ANOVAKernel: public DotKernel
{
public:
	/** default constructor */
	ANOVAKernel();

	/** constructor
	 * @param cache size of cache
	 * @param d kernel parameter cardinality
	 */
	ANOVAKernel(int32_t cache, int32_t d);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param d kernel parameter cardinality
	 * @param cache cache size
	 */
	ANOVAKernel(
		const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r, int32_t d, int32_t cache);

	virtual ~ANOVAKernel();

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

	/**
	 * @return kernel type
	 */
	virtual EKernelType get_kernel_type() { return K_ANOVA; }

	/**
	 * @return type of features
	 */
	virtual EFeatureType get_feature_type() { return F_DREAL; }

	/**
	 * @return class of features
	 */
	virtual EFeatureClass get_feature_class() { return C_DENSE; }

	/**
	 * @return name of kernel
	 */
	virtual const char* get_name() const { return "ANOVAKernel"; }

	/** getter for degree parameter
	 *  @return kernel parameter cardinality
	 */
	inline int32_t get_cardinality() { return this->cardinality; }

	/** setter for degree parameter
	 *  @param value kernel parameter cardinality
	 */
	inline void set_cardinality(int32_t value) { this->cardinality = value; }

	/** compute rec 1
	 * @param idx_a
	 * @param idx_b
	 * @return rec1
	 */
	float64_t compute_rec1(int32_t idx_a, int32_t idx_b);

	/** computer rec 2
	 * @param idx_a
	 * @param idx_b
	 * @return rec2
	 */
	float64_t compute_rec2(int32_t idx_a, int32_t idx_b);

	/** Casts the given kernel to CANOVAKernel.
	 * @param kernel Kernel to cast. Must be CANOVAKernel. Might be NULL
	 * @return casted CANOVAKernel object, NULL if input was NULL
	 */
	static std::shared_ptr<ANOVAKernel> obtain_from_generic(const std::shared_ptr<Kernel>& kernel);
protected:

	/**
	 * compute kernel for specific feature vectors
	 * corresponding to [idx_a] of left-side and [idx_b] of right-side
	 * @param idx_a left-side index
	 * @param idx_b right-side index
	 * @return kernel value
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	/** register params */
	void register_params();

private:
	float64_t compute_recursive1(float64_t* avec, float64_t* bvec, int32_t len);
	float64_t compute_recursive2(float64_t* avec, float64_t* bvec, int32_t len);

protected:
	/// degree parameter of kernel
	int32_t cardinality;
};
}

#endif /* ANOVAKERNEL_H_ */
