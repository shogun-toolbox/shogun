/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _SPECTRUMRBFKERNEL_H___
#define _SPECTRUMRBFKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/Trie.h>
#include <shogun/kernel/string/StringKernel.h>
#include <shogun/features/StringFeatures.h>


#include <shogun/lib/DynamicArray.h>

#include <vector> // profile
#include <string> // profile

namespace shogun
{

/** @brief spectrum rbf kernel */
class SpectrumRBFKernel: public StringKernel<char>
{
	public:
		/** default constructor  */
		SpectrumRBFKernel();

		/** constructor
		 * @param size
		 * @param AA_matrix
		 * @param degree
		 * @param width
		 */
		SpectrumRBFKernel(int32_t size, float64_t* AA_matrix, int32_t degree, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param size
		 * @param AA_matrix
		 * @param degree
		 * @param width
		 */
		SpectrumRBFKernel(
			const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r, int32_t size, float64_t* AA_matrix, int32_t degree, float64_t width);

		/** destructor */
		~SpectrumRBFKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** clean up kernel */
		void cleanup() override;

		/** get degree
		 *
		 * @return degree of the kernel
		 */
		int32_t get_degree() const
		{
			return degree;
		}

		/** return what type of kernel we are
		 *
		 * @return kernel type
		 */
		EKernelType get_kernel_type() override { return K_SPECTRUMRBF; }

		/** return the kernel's name
		 *
		 * @return name
		 */
		const char* get_name() const override { return "SpectrumRBFKernel"; }

		/** set degree
		 *
		 * @param deg new degree
		 * @return if setting was successful
		 */
		inline bool set_degree(int32_t deg) { degree=deg; return true; }

		/** get degree
		 *
		 * @return degree
		 */
		inline int32_t get_degree() { return degree; }

		/** set AA matrix
		 * @param AA_matrix_
		 */
		bool set_AA_matrix(float64_t* AA_matrix_);

	protected:

		/** AA helper
		 * @param path
		 * @param degree
		 * @param joint_seq
		 * @param index
		 */
		float64_t AA_helper(const char* path, const int degree, const char* joint_seq, unsigned int index);

		/** read profiles and sequences */
		void read_profiles_and_sequences();

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b) override;

		/** register the parameters */
		virtual void register_param();
		/** register the alphabet */
		void register_alphabet();


	protected:
		/** alphabet of features */
		std::shared_ptr<Alphabet> alphabet;
		/** degree */
		int32_t degree;
		/** maximum mismatch */
		int32_t max_mismatch;
		/**  128x128 scalar product matrix */
		SGMatrix<float64_t> AA_matrix ;
		/** width of Gaussian*/
		float64_t width;

		//int32_t* aa_to_index; // profile

		//double background[20]; // profile
		/** profiles */
		std::vector< std::vector<float64_t> > profiles; //profile
		/** sequence labels */
		std::vector<std::string> sequence_labels; // profile
		/** sequences */
		SGVector<char>* sequences; // profile
		/** string features */
		std::shared_ptr<StringFeatures<char>> string_features;
		/** nof sequences */
		int32_t nof_sequences;
		/** max sequence length */
		int32_t max_sequence_length;

		/** if kernel is initialized */
		bool initialized;
		/** kernel matrix */
		DynamicArray<float64_t> kernel_matrix; // 2d
		/** target letter 0 */
		int32_t target_letter_0;

	private:
		void init();
};

}



#endif /* _SPECTRUMMISMATCHRBFKERNEL_H__ */
