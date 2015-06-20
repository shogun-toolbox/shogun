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
	class CDotFeatures;

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

class CPeriodicKernel: public CDotKernel
{
	public:
		/** default constructor */
		CPeriodicKernel();

		/** constructor
		 *
		 * @param length_scale length_scale
		 * @param period period
		 * @param size cache size. Default value: 10
		 */
		CPeriodicKernel(float64_t length_scale, float64_t period, int32_t size=10);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param length_scale length scale
		 * @param period period
		 * @param size cache size. Default value: 10
		 */
		CPeriodicKernel(CDotFeatures* l, CDotFeatures* r,
			float64_t length_scale, float64_t period, int32_t size=10);

		virtual ~CPeriodicKernel() { };

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** return what type of kernel we are
		 *
		 * @return kernel type PERIODIC
		 */
		virtual EKernelType get_kernel_type() { return K_PERIODIC; }

		/** return the kernel's name
		 *
		 * @return name PeriodicKernel
		 */
		virtual const char* get_name() const { return "PeriodicKernel"; }

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
		virtual SGMatrix<float64_t> get_parameter_gradient(
			const TParameter* param, index_t index=-1);

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);


		/** compute the euclidean distance between features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object.
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed the distance
		 *
		 * Note that this function is very similar the distance function in
		 * CGaussianKernel. However, this function computes the standard
		 * euclidean distance without any factors or squaring, unlike the one
		 * in CGaussianKernel.
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
		void precompute_squared_helper(SGVector<float64_t>& buf, CDotFeatures* df);

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
