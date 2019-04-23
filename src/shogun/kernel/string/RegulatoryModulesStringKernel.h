/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _REGULATORYMODULESSTRINGKERNEL_H___
#define _REGULATORYMODULESSTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief The Regulaty Modules kernel, based on the WD kernel,
 * as published in Schultheiss et al., Bioinformatics (2009)
 * on regulatory sequences.
 *
 */
class RegulatoryModulesStringKernel: public StringKernel<char>
{
	public:
		/** default constructor  */
		RegulatoryModulesStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param width width of gaussian kernel
		 * @param degree degree of wds kernel
		 * @param shift shift of wds kernel
		 * @param window size of window around motifs to compute wds kernels on
		 */
		RegulatoryModulesStringKernel(int32_t size, float64_t width, int32_t degree, int32_t shift, int32_t window);

		/** constructor
		 *
		 * @param lstr string features of left-hand side
		 * @param rstr string features of right-hand side
		 * @param lpos motif positions on lhs
		 * @param rpos motif positions on rhs
		 * @param width width of gaussian kernel
		 * @param degree degree of wds kernel
		 * @param shift shift of wds kernel
		 * @param window size of window around motifs to compute wds kernels on
		 * @param size cache size
		 */
		RegulatoryModulesStringKernel(std::shared_ptr<StringFeatures<char>> lstr, std::shared_ptr<StringFeatures<char>> rstr,
			std::shared_ptr<DenseFeatures<uint16_t>> lpos, std::shared_ptr<DenseFeatures<uint16_t>> rpos,
			float64_t width, int32_t degree, int32_t shift, int32_t window, int32_t size=10);

		/** default destructor */
		virtual ~RegulatoryModulesStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** return what type of kernel we are
		 *
		 * @return kernel type
		 */
		virtual EKernelType get_kernel_type() { return K_REGULATORYMODULES; }

		/** return the kernel's name
		 *
		 * @return name Regulatory Modules
		 */
		virtual const char* get_name() const { return "RegulatoryModulesStringKernel"; }

		/** set motif positions
		 *
		 * @param positions_lhs motif positions on lhs
		 * @param positions_rhs motif positions on rhs
		 */
		void set_motif_positions(
			std::shared_ptr<DenseFeatures<uint16_t>> positions_lhs, std::shared_ptr<DenseFeatures<uint16_t>> positions_rhs);

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

		/** compute WDS kernel for features a and b
		 *
		 * @param avec vector a
		 * @param bvec vector b
		 * @param len length of string
		 * @return computed kernel function
		 */
		float64_t compute_wds(char* avec, char* bvec, int32_t len);


		/** set standard weighted degree kernel weighting */
		void set_wd_weights();

	private:
		/** initialises parameters and registers them */
		void init();

	protected:
		/** width of Gaussian kernel part */
		float64_t width;

		/** degree of Weighted Degree kernel part */
		int32_t degree;
		/** shift of Weighted Degree with Shifts kernel part */
		int32_t shift;

		/** size of window around motifs */
		int32_t window;

		/** Matrix of motif positions from sequences left-hand side */
		std::shared_ptr<DenseFeatures<uint16_t>> motif_positions_lhs;

		/** Matrix of motif positions from sequences right-hand side */
		std::shared_ptr<DenseFeatures<uint16_t>> motif_positions_rhs;

		/** scaling weights in window */
		SGVector<float64_t> position_weights;

		/** weights of WD kernel */
		SGVector<float64_t> weights;
};
}
#endif /* _REGULATORYMODULESSTRINGKERNEL_H__ */
