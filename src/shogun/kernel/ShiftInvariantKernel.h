/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/lib/config.h>

#ifndef SHIFT_INVARIANT_KERNEL_H_
#define SHIFT_INVARIANT_KERNEL_H_

#include <shogun/kernel/Kernel.h>
#include <shogun/distance/CustomDistance.h>

namespace shogun
{

namespace internal
{
	class KernelManager;
}

/** @brief Base class for the family of kernel functions that only depend on
 * the difference of the inputs, i.e. whose values does not change if the
 * inputs are shifted by the same amount. More precisely,
 * \f[
 * 	k(\mathbf{x}, \mathbf{x'}) = k(\mathbf{x-x'})
 * \f]
 * For example, Gaussian (RBF) kernel is a shfit invariant kernel.
 */
class ShiftInvariantKernel: public Kernel
{

	friend class internal::KernelManager;

public:
	/** Default constructor.  */
	ShiftInvariantKernel();

	/**
	 * Constructor that initializes the kernel with two feature instances.
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 */
	ShiftInvariantKernel(const std::shared_ptr<Features >&l, const std::shared_ptr<Features >&r);

	/** Destructor. */
	~ShiftInvariantKernel() override;

	/**
	 * Initialize kernel.
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if initializing was successful
	 */
	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

	/** Method that precomputes the distance */
	virtual void precompute_distance();

	/**
	 * Method that releases any precomputed distance instance in addition to
	 * clean up the base class methods.
	 */
	void cleanup() override;

	/** compute kernel function for features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * abstract base method
	 *
	 * @param x index a
	 * @param y index b
	 * @return computed kernel function at indices a,b
	 */
	float64_t compute(int32_t x, int32_t y) override =0;

	/** @return kernel type */
	EKernelType get_kernel_type() override =0;

	/** @return feature type of distance used */
	EFeatureType get_feature_type() override =0;

	/** @return feature class of distance used */
	EFeatureClass get_feature_class() override =0;

	/** @return the distance type */
	virtual EDistanceType get_distance_type() const;

	/** @return name Distance */
	const char* get_name() const override
	{
		return "ShiftInvariantKernel";
	}

protected:
	/**
	 * Computes distance between features a and b, where idx_{a,b} denote the indices
	 * of the feature vectors in the corresponding feature object.
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return distance between features a and b
	 */
	virtual float64_t distance(int32_t idx_a, int32_t idx_b) const;

	/** Distance instance for the kernel. MUST be initialized by the subclasses */
	std::shared_ptr<Distance> m_distance;

private:
	/** Registers the parameters (serialization support). */
	void register_params() override;

	/** Precomputed distance instance */
	std::shared_ptr<CustomDistance> m_precomputed_distance;

	/**
	 * Method that sets a precomputed distance.
	 *
	 * @param precomputed_distance The precomputed distance object.
	 */
	void set_precomputed_distance(const std::shared_ptr<CustomDistance>& precomputed_distance);

	/** @return the precomputed distance. */
	std::shared_ptr<CustomDistance> get_precomputed_distance() const;

};

}

#endif // SHIFT_INVARIANT_KERNEL_H__
