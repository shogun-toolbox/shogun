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

#ifndef _SPECTRUMMISMATCHRBFKERNEL_H___
#define _SPECTRUMMISMATCHRBFKERNEL_H___

#include "lib/common.h"
#include "lib/Trie.h"
#include "kernel/StringKernel.h"
#include "features/StringFeatures.h"


#include "lib/Array.h"
#include "lib/Array2.h"
#include <string>

namespace shogun
{

struct joint_list_struct
{
	unsigned int ex_index ;
	unsigned int index ;
	unsigned int mismatch ;
} ;

class CSpectrumMismatchRBFKernel: public CStringKernel<char>
{
	public:
		/** default constructor  */
		CSpectrumMismatchRBFKernel(void);

		/** constructor
		 *
		 * @param 
		 * @param degree degree
		 */
		CSpectrumMismatchRBFKernel(int32_t size, float64_t* AA_matrix_, int32_t nr_, int32_t nc_, int32_t degree, int32_t max_mismatch, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 */
		CSpectrumMismatchRBFKernel(
                                   CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t size, float64_t* AA_matrix_, int32_t nr_, int32_t nc_, int32_t degree, int32_t max_mismatch, float64_t width);

		virtual ~CSpectrumMismatchRBFKernel();

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

		/** get the number of mismatches that are allowed
		 *
		 * @return number of mismatches
		 */
		int32_t get_max_mismatch() const
		{
			return max_mismatch;
		}

		/** return what type of kernel we are
		 *
		 * @return kernel type 
		 */
		virtual EKernelType get_kernel_type() { return K_SPECTRUMMISMATCHRBF; }

		/** return the kernel's name
		 *
		 * @return name 
		 */
		virtual const char* get_name() const { return "SpectrumMismatchRBF"; }

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
		inline int32_t get_max_mismatch() { return max_mismatch; }

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


		bool set_AA_matrix(float64_t* AA_matrix_=NULL, int32_t nr=128, int32_t nc=128);

	protected:

		float64_t AA_helper(std::string &path, const char* joint_seq, unsigned int index) ;
		float64_t compute_helper(const char* joint_seq, 
								 std::vector<unsigned int> joint_index, std::vector<unsigned int> joint_mismatch, 
								 std::string path, unsigned int d, 
								 const int & alen) ;

		
		void compute_helper_all(const char* joint_seq, 
								std::vector<struct joint_list_struct> & joint_list,
								std::string path, unsigned int d)  ;
		void compute_all() ;
		

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


	protected:
		/** alphabet of features */
		CAlphabet* alphabet;
		/** degree */
		int32_t degree;
		/** maximum mismatch */
		int32_t max_mismatch;
		/**  128x128 scalar product matrix */
		float64_t* AA_matrix;
		/** width of Gaussian*/
		float64_t width;

		/** if kernel is initialized */
		bool initialized;


		CArray2<float64_t> kernel_matrix ;
		int32_t target_letter_0 ;
};

}

#endif /* _SPECTRUMMISMATCHRBFKERNEL_H__ */
