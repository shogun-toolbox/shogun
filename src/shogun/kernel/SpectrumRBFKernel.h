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

#include <shogun/lib/common.h>
#include <shogun/lib/Trie.h>
#include <shogun/kernel/StringKernel.h>
#include <shogun/features/StringFeatures.h>


#include <shogun/lib/Array.h>
#include <shogun/lib/Array2.h>

#include <vector> // profile
#include <string> // profile

namespace shogun
{

class CSpectrumRBFKernel: public CStringKernel<char>
{
	public:
		/** default constructor  */
		CSpectrumRBFKernel(void);

		/** constructor
		 *
		 * @param 
		 * @param degree degree
		 */
		CSpectrumRBFKernel(int32_t size, float64_t* AA_matrix, int32_t degree, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 */
		CSpectrumRBFKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t size, float64_t* AA_matrix, int32_t degree, float64_t width);

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


		bool set_AA_matrix(float64_t* AA_matrix_);

	protected:

		float64_t AA_helper(const char* path, const int degree, const char* joint_seq, unsigned int index);

		void read_profiles_and_sequences(); // profile
		

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b);

		/** remove lhs from kernel */
		virtual void remove_lhs();
	    /* register the parameters */
	    virtual void register_param();
		/* register the alphabet */
		void register_alphabet();


	protected:
		/** alphabet of features */
		CAlphabet* alphabet;
		/** degree */
		int32_t degree;
		/** maximum mismatch */
	    int32_t max_mismatch;
		/**  128x128 scalar product matrix */
		float64_t* AA_matrix ; 
	    /*length of the AA_matrix -- for registration*/
	    int32_t AA_matrix_length;
		/** width of Gaussian*/
		float64_t width;

		//int32_t* aa_to_index; // profile

		//double background[20]; // profile
		std::vector< std::vector<float64_t> > profiles; //profile
		std::vector<std::string> sequence_labels; // profile
		SGString<char>* sequences; // profile
		CStringFeatures<char>* string_features; 
		int32_t nof_sequences;
		int32_t max_sequence_length;

		/** if kernel is initialized */
		bool initialized;
		

		CArray2<float64_t> kernel_matrix;
		int32_t target_letter_0;
	
	private:
		void init();
};

}



#endif /* _SPECTRUMMISMATCHRBFKERNEL_H__ */
