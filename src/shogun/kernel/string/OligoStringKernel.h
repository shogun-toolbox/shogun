/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Bjoern Esser, Sergey Lisitsyn
 */
#ifndef _OLIGOSTRINGKERNEL_H_
#define _OLIGOSTRINGKERNEL_H_

#include <shogun/lib/config.h>

#include <shogun/kernel/string/StringKernel.h>

#include <vector>
#include <string>

namespace shogun
{
/**
 * @brief This class offers access to the Oligo Kernel introduced
 * by Meinicke et al. in 2004
 *
 * The class has functions to preprocess the data such that the kernel
 * computation can be pursued faster. The kernel function is then
 * kernelOligoFast or kernelOligo.
 *
 * Requires significant speedup, should be working but as is might be
 * applicable only to academic small scale problems:
 *
 * - the kernel should only ever see encoded sequences, which however
 * requires another OligoFeatures object (using DenseFeatures of pairs)
 *
 * Uses CSqrtDiagKernelNormalizer, as the vanilla kernel seems to be very
 * diagonally dominant.
 *
 */
class OligoStringKernel : public StringKernel<char>
{
	public:
		/** default constructor  */
		OligoStringKernel();

		/** Constructor
		 * @param cache_size cache size for kernel
		 * @param k k-mer length
		 * @param width - equivalent to 2*sigma^2
		 */
		OligoStringKernel(int32_t cache_size, int32_t k, float64_t width);

		/** Constructor
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param k k-mer length
		 * @param width - equivalent to 2*sigma^2
		 */
		OligoStringKernel(
				const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r,
				int32_t k, float64_t width);

		/** Destructor */
		virtual ~OligoStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** return what type of kernel we are
		 *
		 * @return kernel type OLIGO
		 */
		virtual EKernelType get_kernel_type() { return K_OLIGO; }

		/** return the kernel's name
		 *
		 * @return name Oligo
		 */
		virtual const char* get_name() const { return "OligoStringKernel"; }


		virtual float64_t compute(int32_t x, int32_t y);

		/** clean up your kernel
		 */
		virtual void cleanup();

	protected:
		/**
		 * @brief encodes the signals of the sequence
		 *
		 * This function stores the oligo function signals in 'values'.
		 *
		 * The 'k_mer_length' and the 'allowed_characters' determine,
		 * which signals are used. Every pair contains the position of the
		 * signal and a numerical value reflecting the signal. The
		 * numerical value represents the k_mer to a base
		 * n = |allowed_characters|.
		 * Example: The value of k_mer CG for the allowed characters ACGT
		 * would be 1 * n^1 + 2 * n^0 = 6.
		 */
		static void encodeOligo(
			const std::string& sequence, uint32_t k_mer_length,
			const std::string& allowed_characters,
			std::vector< std::pair<int32_t, float64_t> >&   values);

		/**
		  @brief encodes all sequences with the encodeOligo function and stores
		  them in 'encoded_sequences'

		  This function encodes the sequences of 'sequences' via the
		  function encodeOligo.
		  */
		static void getSequences(
			const std::vector<std::string>& sequences,
			uint32_t k_mer_length, const std::string& allowed_characters,
			std::vector< std::vector< std::pair<int32_t, float64_t> > >& encoded_sequences);

		/**
		  @brief returns the value of the oligo kernel for sequences 'x' and 'y'

		  This function computes the kernel value of the oligo kernel,
		  which was introduced by Meinicke et al. in 2004. 'x' and
		  'y' are encoded by encodeOligo and 'exp_cache' has to be
		  constructed by getExpFunctionCache.

		  'max_distance' can be used to speed up the computation
		  even further by restricting the maximum distance between a k_mer at
		  position i in sequence 'x' and a k_mer at position j
		  in sequence 'y'. If i - j > 'max_distance' the value is not
		  added to the kernel value. This approximation is switched
		  off by default (max_distance < 0).
		  */
		float64_t kernelOligoFast(
			const std::vector< std::pair<int32_t, float64_t> >& x,
			const std::vector< std::pair<int32_t, float64_t> >& y,
			int32_t max_distance = -1);

		/**
		  @brief returns the value of the oligo kernel for sequences 'x' and 'y'

		  This function computes the kernel value of the oligo kernel,
		  which was introduced by Meinicke et al. in 2004. 'x' and
		  'y' have to be encoded by encodeOligo.
		  */
		float64_t kernelOligo(
				const std::vector< std::pair<int32_t, float64_t> >& x,
				const std::vector< std::pair<int32_t, float64_t> >& y);


	private:
		/**
		  @brief prepares the exp function cache of the oligo kernel

		  The oligo kernel was introduced for sequences of fixed length.
		  Let n be the sequence length of sequences x and y. There can
		  only be n different distances between signals in sequence x
		  and sequence y (0, 1, ..., n-1). Therefore, we precompute
		  the corresponding values of the e-function. These values
		  can then be used in kernelOligoFast.
		  */
		void getExpFunctionCache(uint32_t sequence_length);

		static inline bool cmpOligos_(std::pair<int32_t, float64_t> a,
				std::pair<int32_t, float64_t> b )
		{
			return (a.second < b.second);
		}

		void init();

	protected:
		/** k-mer length */
		int32_t k;
		/** width of kernel */
		float64_t width;
		/** gauss table cache for exp (see getExpFunctionCache above) */
		SGVector<float64_t> gauss_table;
};
}
#endif // _OLIGOSTRINGKERNEL_H_
