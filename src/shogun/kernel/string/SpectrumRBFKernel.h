/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPECTRUMRBFKERNEL_H___
#define _SPECTRUMRBFKERNEL_H___

#include <lib/common.h>
#include <lib/Trie.h>
#include <kernel/string/StringKernel.h>
#include <features/StringFeatures.h>


#include <lib/DynamicArray.h>

#include <vector> // profile
#include <string> // profile

namespace shogun
{

/** @brief spectrum rbf kernel */
class CSpectrumRBFKernel: public CStringKernel<char>
{
	public:
		/** default constructor  */
		CSpectrumRBFKernel();

		/** constructor
		 * @param size
		 * @param AA_matrix
		 * @param degree
		 * @param width
		 */
		CSpectrumRBFKernel(int32_t size, float64_t* AA_matrix, int32_t degree, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param size
		 * @param AA_matrix
		 * @param degree
		 * @param width
		 */
		CSpectrumRBFKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t size, float64_t* AA_matrix, int32_t degree, float64_t width);

		/** destructor */
		virtual ~CSpectrumRBFKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

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
		virtual EKernelType get_kernel_type() { return K_SPECTRUMRBF; }

		/** return the kernel's name
		 *
		 * @return name
		 */
		virtual const char* get_name() const { return "SpectrumRBFKernel"; }

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
		float64_t compute(int32_t idx_a, int32_t idx_b);

		/** register the parameters */
		virtual void register_param();
		/** register the alphabet */
		void register_alphabet();


	protected:
		/** alphabet of features */
		CAlphabet* alphabet;
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
		SGString<char>* sequences; // profile
		/** string features */
		CStringFeatures<char>* string_features;
		/** nof sequences */
		int32_t nof_sequences;
		/** max sequence length */
		int32_t max_sequence_length;

		/** if kernel is initialized */
		bool initialized;
		/** kernel matrix */
		CDynamicArray<float64_t> kernel_matrix; // 2d
		/** target letter 0 */
		int32_t target_letter_0;

	private:
		void init();
};

}



#endif /* _SPECTRUMMISMATCHRBFKERNEL_H__ */
