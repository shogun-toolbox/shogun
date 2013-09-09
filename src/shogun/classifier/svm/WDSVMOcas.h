/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2008 Vojtech Franc
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WDSVMOCAS_H___
#define _WDSVMOCAS_H___

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/labels/Labels.h>

namespace shogun
{
template <class ST> class CStringFeatures;

/** @brief class WDSVMOcas */
class CWDSVMOcas : public CMachine
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** default constructor  */
		CWDSVMOcas();

		/** constructor
		 *
		 * @param type type of SVM
		 */
		CWDSVMOcas(E_SVM_TYPE type);

		/** constructor
		 *
		 * @param C constant C
		 * @param d degree
		 * @param from_d from degree
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CWDSVMOcas(
			float64_t C, int32_t d, int32_t from_d,
			CStringFeatures<uint8_t>* traindat, CLabels* trainlab);
		virtual ~CWDSVMOcas();

		/** get classifier type
		 *
		 * @return classifier type WDSVMOCAS
		 */
		virtual EMachineType get_classifier_type() { return CT_WDSVMOCAS; }

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

		/** set features
		 *
		 * @param feat features to set
		 */
		inline void set_features(CStringFeatures<uint8_t>* feat)
		{
			SG_REF(feat);
			SG_UNREF(features);
			features=feat;
		}

		/** get features
		 *
		 * @return features
		 */
		inline CStringFeatures<uint8_t>* get_features()
		{
			SG_REF(features);
			return features;
		}

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

		/** set degree
		 *
		 * @param d degree
		 * @param from_d from degree
		 */
		inline void set_degree(int32_t d, int32_t from_d)
		{
			degree=d;
			from_degree=from_d;
		}

		/** get degree
		 *
		 * @return degree
		 */
		inline int32_t get_degree() { return degree; }

		/** classify objects
		 * for binary classification problems
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CBinaryLabels* apply_binary(CFeatures* data=NULL);
		
		/** classify objects
		 * for regression problems
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);

		/** classify one example
		 *
		 * @param num number of example to classify
		 * @return classified result
		 */
		virtual float64_t apply_one(int32_t num)
		{
			ASSERT(features)
			if (!wd_weights)
				set_wd_weights();

			int32_t len=0;
			float64_t sum=0;
			bool free_vec;
			uint8_t* vec=features->get_feature_vector(num, len, free_vec);
			//SG_INFO("len %d, string_length %d\n", len, string_length)
			ASSERT(len==string_length)

			for (int32_t j=0; j<string_length; j++)
			{
				int32_t offs=w_dim_single_char*j;
				int32_t val=0;
				for (int32_t k=0; (j+k<string_length) && (k<degree); k++)
				{
					val=val*alphabet_size + vec[j+k];
					sum+=wd_weights[k] * w[offs+val];
					offs+=w_offsets[k];
				}
			}
			features->free_feature_vector(vec, num, free_vec);
			return sum/normalization_const;
		}

		/** set normalization const */
		inline void set_normalization_const()
		{
			ASSERT(features)
			normalization_const=0;
			for (int32_t i=0; i<degree; i++)
				normalization_const+=(string_length-i)*wd_weights[i]*wd_weights[i];

			normalization_const=CMath::sqrt(normalization_const);
			SG_DEBUG("normalization_const:%f\n", normalization_const)
		}

		/** get normalization const
		 *
		 * @return normalization const
		 */
		inline float64_t get_normalization_const() { return normalization_const; }


	protected:

		/** get real outputs
		 *
		 * @param data features to apply for
		 */
		SGVector<float64_t> apply_get_outputs(CFeatures* data);

		/** set wd weights
		 *
		 * @return w_dim_single_c
		 */
		int32_t set_wd_weights();

		/** compute W
		 *
		 * @param sq_norm_W square normed W
		 * @param dp_WoldW dp W old W
		 * @param alpha alpha
		 * @param nSel nSel
		 * @param ptr ptr
		 */
		static void compute_W(
			float64_t *sq_norm_W, float64_t *dp_WoldW, float64_t *alpha,
			uint32_t nSel, void* ptr );

		/** update W
		 *
		 * @param t t
		 * @param ptr ptr
		 * @return something floaty
		 */
		static float64_t update_W(float64_t t, void* ptr );

		/** helper function for adding a new cut
		 *
		 * @param ptr
		 * @return ptr
		 */
		static void* add_new_cut_helper(void* ptr);

		/** add new cut
		 *
		 * @param new_col_H new col H
		 * @param new_cut new cut
		 * @param cut_length length of cut
		 * @param nSel nSel
		 * @param ptr ptr
		 */
		static int add_new_cut(
			float64_t *new_col_H, uint32_t *new_cut, uint32_t cut_length,
			uint32_t nSel, void* ptr );

		/** helper function for computing the output
		 *
		 * @param ptr
		 * @return ptr
		 */
		static void* compute_output_helper(void* ptr);

		/** compute output
		 *
		 * @param output output
		 * @param ptr ptr
		 */
		static int compute_output( float64_t *output, void* ptr );

		/** sort
		 *
		 * @param vals vals
		 * @param data data
		 * @param size size
		 */
		static int sort( float64_t* vals, float64_t* data, uint32_t size);

		/** print nothing */
		static inline void print(ocas_return_value_T value)
		{
			  return;
		}


		/** @return object name */
		virtual const char* get_name() const { return "WDSVMOcas"; }

	protected:
		/** train classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	protected:
		/** features */
		CStringFeatures<uint8_t>* features;
		/** if bias shall be used */
		bool use_bias;
		/** buffer size */
		int32_t bufsize;
		/** C1 */
		float64_t C1;
		/** C2 */
		float64_t C2;
		/** epsilon */
		float64_t epsilon;
		/** method */
		E_SVM_TYPE method;

		/** degree */
		int32_t degree;
		/** from degree */
		int32_t from_degree;
		/** wd weights */
		float32_t* wd_weights;
		/** num vectors */
		int32_t num_vec;
		/** length of string in vector */
		int32_t string_length;
		/** size of alphabet */
		int32_t alphabet_size;

		/** normalization const */
		float64_t normalization_const;

		/** bias */
		float64_t bias;
		/** old_bias */
		float64_t old_bias;
		/** w offsets */
		int32_t* w_offsets;
		/** w dim */
		int32_t w_dim;
		/** w dim of a single char */
		int32_t w_dim_single_char;
		/** w */
		float32_t* w;
		/** old w*/
		float32_t* old_w;
		/** labels */
		float64_t* lab;

		/** cuts */
		float32_t** cuts;
		/** bias dimensions */
		float64_t* cp_bias;
};
}
#endif
