/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _SNPSTRINGKERNEL_H___
#define _SNPSTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/memory.h>
#include <shogun/kernel/string/StringKernel.h>

namespace shogun
{
/** @brief The class SNPStringKernel computes a variant of the polynomial
 * kernel on strings of same length.
 *
 * It is computed as FIXME
 *
 * \f[
 * k({\bf x},{\bf x'})= (\sum_{i=0}^{L-1} I(x_i=x'_i)+c)^d
 * \f]
 *
 * where I is the indicator function which evaluates to 1 if its argument is
 * true and to 0 otherwise.
 *
 * Note that additional normalisation is applied, i.e.
 * \f[
 *     k'({\bf x}, {\bf x'})=\frac{k({\bf x}, {\bf x'})}{\sqrt{k({\bf x}, {\bf x})k({\bf x'}, {\bf x'})}}
 * \f]
 */
class SNPStringKernel: public StringKernel<char>
{
	public:
		/** default constructor  */
		SNPStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param degree degree
		 * @param win_len length of local window
		 * @param inhomogene whether inhomogeneous poly
		 */
		SNPStringKernel(int32_t size, int32_t degree, int32_t win_len, bool inhomogene);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 * @param win_len length of local window
		 * @param inhomogene whether inhomogeneous poly
		 */
		SNPStringKernel(
			const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r,
			int32_t degree, int32_t win_len, bool inhomogene);

		~SNPStringKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** clean up kernel */
		void cleanup() override;

		/** return what type of kernel we are
		 *
		 * @return kernel type POLYMATCH
		 */
		EKernelType get_kernel_type() override
		{
			return K_POLYMATCH;
		}

		/** set the base string for minor aleles
		 *
		 * @param str minor freq. string
		 */
		void set_minor_base_string(const char* str)
		{
			m_str_min=get_strdup(str);
		}

		/** set the base string for major aleles
		 *
		 * @param str major freq. string
		 */
		void set_major_base_string(const char* str)
		{
			m_str_maj=get_strdup(str);
		}

		/** get the base string for minor aleles
		 *
		 * @return minor freq. string
		 */
		char* get_minor_base_string()
		{
			return m_str_min;
		}

		/** get the base string for major aleles
		 *
		 * @return major freq. string
		 */
		char* get_major_base_string()
		{
			return m_str_maj;
		}

		/** compute the minor / major alele base strings */
		void obtain_base_strings();

		/** return the kernel's name
		 *
		 * @return name PolyMatchString
		 */
		const char* get_name() const override { return "SNPStringKernel"; }

		/* register the parameters
		 */
		void register_params() override;

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

	protected:
		/** degree */
		int32_t m_degree;
		/** window length */
		int32_t m_win_len;

		/** inhomogeneous poly kernel ? */
		bool m_inhomogene;

		/** total string length / must match length of min/maj strings and
		 * string length of each vector */
		int32_t m_str_len;

		/** allele A */
		char* m_str_min;
		/** allele B */
		char* m_str_maj;

	private:
		void init();
};
}
#endif /* _SNPSTRINGKERNEL_H___ */
