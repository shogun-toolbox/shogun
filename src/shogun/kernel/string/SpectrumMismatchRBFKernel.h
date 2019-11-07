/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _SPECTRUMMISMATCHRBFKERNEL_H___
#define _SPECTRUMMISMATCHRBFKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/Trie.h>
#include <shogun/kernel/string/StringKernel.h>
#include <shogun/features/StringFeatures.h>

#include <shogun/lib/DynamicArray.h>
#include <string>
#include <vector>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** joint list struct */
struct joint_list_struct
{
	/** ex index */
	unsigned int ex_index;
	/** index */
	unsigned int index;
	/** mismatch */
	unsigned int mismatch;
};
#endif

/** @brief spectrum mismatch rbf kernel */
class SpectrumMismatchRBFKernel: public StringKernel<char>
{
public:
	/** default constructor  */
	SpectrumMismatchRBFKernel();

	/** constructor
	 *
	 * @param size
	 * @param AA_matrix_
	 * @param nr_
	 * @param nc_
	 * @param degree
	 * @param max_mismatch
	 * @param width
	 */
	SpectrumMismatchRBFKernel(int32_t size, float64_t* AA_matrix_, int32_t nr_,
			int32_t nc_, int32_t degree, int32_t max_mismatch, float64_t width);

	/** constructor
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @param size
	 * @param AA_matrix_
	 * @param nr_
	 * @param nc_
	 * @param degree
	 * @param max_mismatch
	 * @param width
	 */
	SpectrumMismatchRBFKernel(const std::shared_ptr<StringFeatures<char>>& l,
			const std::shared_ptr<StringFeatures<char>>& r, int32_t size, float64_t* AA_matrix_,
			int32_t nr_, int32_t nc_, int32_t degree, int32_t max_mismatch,
			float64_t width);

	/** destructor */
	virtual ~SpectrumMismatchRBFKernel();

	/** initialize kernel
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if initializing was successful
	 */
	virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

	/** clean up kernel */
	virtual void cleanup();

	/** return what type of kernel we are
	 *
	 * @return kernel type
	 */
	virtual EKernelType get_kernel_type()
	{
		return K_SPECTRUMMISMATCHRBF;
	}

	/** return the kernel's name
	 *
	 * @return name
	 */
	virtual const char* get_name() const
	{
		return "SpectrumMismatchRBFKernel";
	}

	/** set maximum mismatch
	 *
	 * @param max new maximum mismatch
	 * @return if setting was successful
	 */
	bool set_max_mismatch(int32_t max);

	/** get maximum mismatch
	 *
	 * @return maximum mismatch
	 */
	inline int32_t get_max_mismatch() const
	{
		return max_mismatch;
	}

	/** set degree
	 *
	 * @param deg new degree
	 * @return if setting was successful
	 */
	inline bool set_degree(int32_t deg)
	{
		degree=deg;
		return true;
	}

	/** get degree
	 *
	 * @return degree
	 */
	inline int32_t get_degree() const
	{
		return degree;
	}

	/** set AA matrix
	 * @param AA_matrix_
	 * @param nr
	 * @param nc
	 * @return true if set
	 */
	bool set_AA_matrix(float64_t* AA_matrix_=NULL, int32_t nr=128, int32_t nc=
			128);

protected:

	/** AA helper
	 * @param path
	 * @param joint_seq
	 * @param index
	 * @return AA helper
	 */
	float64_t AA_helper(std::string &path, const char* joint_seq,
			unsigned int index);

	/** compute helper
	 * @param joint_seq
	 * @param joint_index
	 * @param joint_mismatch
	 * @param path
	 * @param d
	 * @param alen
	 * @return helper
	 */
	float64_t compute_helper(const char* joint_seq,
			std::vector<unsigned int> joint_index,
			std::vector<unsigned int> joint_mismatch, std::string path,
			unsigned int d, const int & alen);

	/** compute helper all
	 * @param joint_seq
	 * @param joint_list
	 * @param path
	 * @param d
	 * @return helper
	 */
	void compute_helper_all(const char* joint_seq,
			std::vector<struct joint_list_struct> & joint_list,
			const std::string& path, unsigned int d);

	/** computer all */
	void compute_all();

	/** compute kernel function for features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return computed kernel function at indices a,b
	 */
	float64_t compute(int32_t idx_a, int32_t idx_b);

	/** register the parameters
	 */
	virtual void register_params();
	/** register the alphabet
	 */
	void register_alphabet();

protected:
	/** alphabet of features */
	std::shared_ptr<Alphabet> alphabet;
	/** degree */
	int32_t degree;
	/** maximum mismatch */
	int32_t max_mismatch;
	/**  128x128 scalar product matrix */
	SGMatrix<float64_t> AA_matrix;
	/** width of Gaussian*/
	float64_t width;

	/** if kernel is initialized */
	bool initialized;

	/** kernel matrix */
	std::shared_ptr<DynamicArray<float64_t>> kernel_matrix; // 2d
	/** kernel matrix length */
	int32_t kernel_matrix_length;
	/** target letter 0 */
	int32_t target_letter_0;

private:
	void init();
};

}

#endif /* _SPECTRUMMISMATCHRBFKERNEL_H__ */
