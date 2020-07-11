/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Esben Soerig
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

#ifndef PERIODICKERNEL_H
#define PERIODICKERNEL_H

#include <shogun/kernel/DotKernel.h>

namespace shogun
{
	class DotFeatures;

/** @brief The periodic kernel as described in <i>The Kernel Cookbook</i>
 * by David Duvenaud: http://people.seas.harvard.edu/~dduvenaud/cookbook/
 *
 * It is computed as
 *
 * \f[
 *   k({\bf x},{\bf x'})= exp(-\frac{2sin^2(\pi|{\bf x}-{\bf x'}|/p)}{l^2})
 * \f]
 *
 * where \f$p\f$ is the period of the kernel and \f$l\f$ is the length scale
 * of the kernel.
 */

class PeriodicKernel: public DotKernel
{
	public:
		/** default constructor */
		PeriodicKernel();

		/** constructor
		 *
		 * @param length_scale length_scale
		 * @param period period
		 * @param size cache size. Default value: 10
		 */
		PeriodicKernel(float64_t length_scale, float64_t period, int32_t size=10);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param length_scale length scale
		 * @param period period
		 * @param size cache size. Default value: 10
		 */
		PeriodicKernel(const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r,
			float64_t length_scale, float64_t period, int32_t size=10);

		~PeriodicKernel() override { };

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** return what type of kernel we are
		 *
		 * @return kernel type PERIODIC
		 */
		EKernelType get_kernel_type() override { return K_PERIODIC; }

		/** return the kernel's name
		 *
		 * @return name PeriodicKernel
		 */
		const char* get_name() const override { return "PeriodicKernel"; }

		/** set the kernel's length scale
		 *
		 * @param length_scale kernel length scale
		 */
		virtual void set_length_scale(float64_t length_scale) { m_length_scale=length_scale; }

		/** return the kernel's length scale
		 *
		 * @return kernel length scale
		 */
		virtual float64_t get_length_scale() const { return m_length_scale;	}

		/** set the kernel's period
		 *
		 * @param period kernel period
		 */
		virtual void set_period(float64_t period) { m_period=period; }

		/** return the kernel's period
		 *
		 * @return kernel perid
		 */
		virtual float64_t get_period() const { return m_period;	}

		/** return derivative with respect to specified parameter
		 *
		 * @param param the parameter
		 * @param index the index of the element if parameter is a vector
		 *
		 * @return gradient with respect to parameter
		 */
		SGMatrix<float64_t> get_parameter_gradient(
			Parameters::const_reference param, index_t index=-1) override;

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b) override;


		/** compute the euclidean distance between features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object.
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed the distance
		 *
		 * Note that this function is very similar the distance function in
		 * GaussianKernel. However, this function computes the standard
		 * euclidean distance without any factors or squaring, unlike the one
		 * in GaussianKernel.
		 */
		virtual float64_t distance(int32_t idx_a, int32_t idx_b);
	private:
		/** helper function to compute quadratic terms in
		 * (a-b)^2 (== a^2+b^2-2ab)
		 */
		void precompute_squared();

		/** helper function to compute quadratic terms in
		 * (a-b)^2 (== a^2+b^2-2ab)
		 *
		 * @param buf buffer to store squared terms
		 * @param df dot feature object based on which k(i,i) is computed
		 * */
		void precompute_squared_helper(SGVector<float64_t>& buf, const std::shared_ptr<DotFeatures>& df);

		void init();

	protected:
		/** length scale */
		float64_t m_length_scale;
		/** period */
		float64_t m_period;
		/** squared left-hand side (each lhs vector dotted with itself) */
		SGVector<float64_t> m_sq_lhs;
		/** squared right-hand side (each rhs vector dotted with itself) */
		SGVector<float64_t> m_sq_rhs;
};
}
#endif /* _PERIODICKERNEL_H__ */
