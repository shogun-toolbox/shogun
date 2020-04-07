/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _SPARSESPATIALSAMPLESTRINGKERNEL_H___
#define _SPARSESPATIALSAMPLESTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>

namespace shogun
{
	/// SSKFeatures
	struct SSKFeatures
	{
		/// features
		int *features;
		/// group
		int *group;
		/// n
		int n;
	};

/** @brief Sparse Spatial Sample String Kernel
 * by Pavel Kuksa <pkuksa@cs.rutgers.edu> and
 * Vladimir Pavlovic <vladimir@cs.rutgers.edu>
 */
class SparseSpatialSampleStringKernel: public StringKernel<char>
{
	public:
		/** constructor
		 */
		SparseSpatialSampleStringKernel();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		SparseSpatialSampleStringKernel(const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r);

		~SparseSpatialSampleStringKernel() override;

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
		 * @return kernel type SPARSESPATIALSAMPLE
		 */
		EKernelType get_kernel_type() override
		{
			return K_SPARSESPATIALSAMPLE;
		}

		/** set d
		 * @param max_distance
		 */
		void set_d(int32_t max_distance)
		{
			ASSERT(d>0)
			d=max_distance;
		}

		/** get d */
		int32_t get_d()
		{
			return d;
		}

		/** set t
		 * @param sequence_length
		 */
		void set_t(int32_t sequence_length)
		{
			ASSERT(t==2 || t==3)
			t=sequence_length;
		}

		/** get t */
		int32_t get_t()
		{
			return t;
		}

		/** return the kernel's name
		 *
		 * @return name SparseSpatialSample
		 */
		const char* get_name() const override { return "SparseSpatialSampleStringKernel"; }

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

		/** extract triple
		 * @param S
		 * @param len
		 * @param nStr
		 * @param d1
		 * @param d2
		 */
		SSKFeatures *extractTriple(int **S, int *len, int nStr, int d1, int d2);
		/** extract double
		 * @param S
		 * @param len
		 * @param nStr
		 * @param d1
		 */
		SSKFeatures *extractDouble(int **S, int *len, int nStr, int d1);
		/** compute double
		 * @param idx_a
		 * @param idx_b
		 */
		void compute_double(int32_t idx_a, int32_t idx_b);
		/** compute triple
		 * @param idx_a
		 * @param idx_b
		 */
		void compute_triple(int32_t idx_a, int32_t idx_b);
		/** makes CNTSRTNA
		 * @param sx
		 * @param k
		 * @param r
		 * @param na
		 */
		int* cntsrtna(int *sx, int k, int r, int na);
		/** count and update
		 * @param outK
		 * @param sx
		 * @param g
		 * @param k
		 * @param r
		 * @param nStr
		 */
		void countAndUpdate(int *outK, int *sx, int *g, int k, int r, int nStr);

	protected:
		/** parameter t of the SSSK denotes how many words are considered in the
		 * sequence (valid are only 2 or 3) */
		int32_t t;

		/** parameter d of the SSSK denotes maximum allowed distance between
		 * words in the sequence */
		int32_t d;

		/** is verbose? */
		bool isVerbose;
};
}
#endif /* _SPARSESPATIALSAMPLESTRINGKERNEL_H___ */
