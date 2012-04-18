/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * LatentSVM.h 
 * Written (W) 2012 Shogun Google Summer of Code Xiangyu
 * Mentor By Alexander and Sonney
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LATENTSVM_H___
#define _LATENTSVM_H___

#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/classifier/svm/libocas.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/Labels.h>

namespace shogun
{

/** @brief class LatentSVM */
class CLatentSVM : public CLinearMachine
{
	public:
		/** default constructor  */
		CLatentSVM();

		/** constructor
		 *
		 * @param C constant C
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CLatentSVM(
			float64_t C, CDotFeatures* traindat,
			CLabels* trainlab,SGVector* HiddenVec);

		virtual ~CLatentSVM();

		/** get classifier type
		 *
		 * @return classifier type CT_SVMLatent
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_SVMLatent; }

		/** set C
		 *
		 * @param c_neg new C constant for negatively labeled examples
		 * @param c_pos new C constant for positively labeled examples
		 *
		 */
		inline void set_C(float64_t c_neg, float64_t c_pos) { C1=c_neg; C2=c_pos; }

		/** get C1
		 *
		 * @return C1
		 */
		inline float64_t get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline float64_t get_C2() { return C2; }

		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(float64_t eps) { epsilon=eps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline float64_t get_epsilon() { return epsilon; }

		/** set if bias shall be enabled
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** check if bias is enabled
		 *
		 * @return if bias is enabled
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** set buffer size
		 *
		 * @param sz buffer size
		 */
		inline void set_bufsize(int32_t sz) { bufsize=sz; }

		/** get buffer size
		 *
		 * @return buffer size
		 */
		inline int32_t get_bufsize() { return bufsize; }

	protected:
		/**Concave-Convex Procedure(CCCP)
		 *
		 * @param init_w init weight
		 * @param final_w
		 */
		static void Concave_Convex_Procedure(float64_t *init_w, float64_t *final_w);
		
		/** computes the linear combination of the SVECTOR
		 * @param vector
		 * @param number of vectors
	     */
		static float64_t* add_list_nn(SGVector *vec, uint32_t totwords);

		/** cutting_plane_algorithm
		 *
		 * @param new_cut new cut
		 * @param cut_length cut_length
		 * @param MAX_ITER length of cut
		 * @param nSel nSel
		 * @param epsilon 
		 * @param HiddenVec Latent
		 * @param ptr ptr
		 */
		float64_t cutting_plane_algorithm(float64_t *new_cut, uint32_t cut_length,
			uint32_t MAX_ITER, float64_t C,SGVector *HiddenVec,void* ptr);

		/** find cutting plane of CCCP algorithm
		 *
		 * @param HiddenVec Latent
		 * @param ptr ptr
		 */
		float64_t* find_cutting_plane(void* ptr, float64_t **fycache);
		
	protected:
		/** train SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

		/** @return object name */
		inline virtual const char* get_name() const { return "SVMLatent"; }
	private:
		void init();

	protected:
		/** if bias is used */
		bool use_bias;
		/** buffer size */
		int32_t bufsize;
		/** C1 */
		float64_t C1;
		/** C2 */
		float64_t C2;
		/** epsilon */
		float64_t epsilon;

		/** old W */
		float64_t* old_w;
		/** old bias */
		float64_t old_bias;
		/** nDim big */
		float64_t* tmp_a_buf;
		/** labels */
		SGVector<float64_t> lab;
		/** Hidden Variables */
		SGVector<float64_t> hiddenvec;

		/** sparse representation of
		 * cutting planes */
		float64_t** cp_value;
		/** cutting plane index */
		uint32_t** cp_index;
		/** cutting plane dimensions */
		uint32_t* cp_nz_dims;
		/** bias dimensions */
		float64_t* cp_bias;

		static float64_t sprod_nn(float64_t *a, float64_t *b, uint32_t n);
		void add_vector_nn(float64_t *w, float64_t *dense_x, uint32_t n, float64_t factor);

		void init_latent_variables(float64_t *sample, void* ptr);
		float64_t *psi(float64_t x, uint32_t y, float64_t h);
		void classify_struct_example(float64_t x, uint32_t *y, float64_t *h, );
		void find_most_violated_constraint_marginrescaling(float64_t x, uint32_t y, 
			float64_t *ybar, float64_t *hbar,void* ptr);
};
}
#endif
