/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2008 Vojtech Franc
 * Written (W) 2007-2008 Soeren Sonnenburg
 * Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WDSVMOCAS_H___
#define _WDSVMOCAS_H___

#include "lib/common.h"
#include "classifier/Classifier.h"
#include "classifier/svm/SVMOcas.h"
#include "features/StringFeatures.h"
#include "features/Labels.h"

/** class WDSVMOcas */
class CWDSVMOcas : public CClassifier
{
	public:
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
		CWDSVMOcas(DREAL C, INT d, INT from_d, CStringFeatures<BYTE>* traindat, CLabels* trainlab);
		virtual ~CWDSVMOcas();

		/** get classifier type
		 *
		 * @return classifier type WDSVMOCAS
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_WDSVMOCAS; }

		/** train SVM
		 *
		 * @return if training was succesful
		 */
		virtual bool train();

		/** set C
		 *
		 * @param c1 new C1
		 * @param c2 new C2
		 */
		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		/** get C1
		 *
		 * @return C1
		 */
		inline DREAL get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline DREAL get_C2() { return C2; }

		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(DREAL eps) { epsilon=eps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline DREAL get_epsilon() { return epsilon; }

		/** set features
		 *
		 * @param feat features to set
		 */
		inline void set_features(CStringFeatures<BYTE>* feat) { features=feat; }

		/** get features
		 *
		 * @return features
		 */
		inline CStringFeatures<BYTE>* get_features() { return features; }

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
		inline void set_bufsize(INT sz) { bufsize=sz; }

		/** get buffer size
		 *
		 * @return buffer size
		 */
		inline INT get_bufsize() { return bufsize; }

		/** set degree
		 *
		 * @param d degree
		 * @param from_d from degree
		 */
		inline void set_degree(INT d, INT from_d) { degree=d; from_degree=from_d;}

		/** get degree
		 *
		 * @return degree
		 */
		inline INT get_degree() { return degree; }

		/** classify all examples
		 *
		 * @param output resulting labels
		 * @return resulting labels
		 */
		CLabels* classify(CLabels* output);

		/** classify one example
		 *
		 * @param num number of example to classify
		 * @return classified result
		 */
		inline virtual DREAL classify_example(INT num)
		{
			ASSERT(features);
			if (!wd_weights)
				set_wd_weights();

			INT len=0;
			DREAL sum=0;
			BYTE* vec = features->get_feature_vector(num, len);
			ASSERT(len == string_length);

			for (INT j=0; j<string_length; j++)
			{
				INT offs=w_dim_single_char*j;
				INT val=0;
				for (INT k=0; (j+k<string_length) && (k<degree); k++)
				{
					val=val*alphabet_size + vec[j+k];
					sum+=wd_weights[k] * w[offs+val];
					offs+=w_offsets[k];
				}
			}
			return sum/normalization_const;
		}

		/** set normalization const */
		inline void set_normalization_const()
		{
			ASSERT(features);
			normalization_const=0;
			for (INT i=0; i<degree; i++)
				normalization_const+=(string_length-i)*wd_weights[i]*wd_weights[i];

			normalization_const=CMath::sqrt(normalization_const);
			SG_DEBUG("normalization_const:%f\n", normalization_const);
		}

		/** get normalization const
		 *
		 * @return normalization const
		 */
		inline DREAL get_normalization_const() { return normalization_const; }


	protected:
		/** set wd weights
		 *
		 * @return something inty
		 */
		INT set_wd_weights();

		/** compute W
		 *
		 * @param sq_norm_W square normed W
		 * @param dp_WoldW dp W old W
		 * @param alpha alpha
		 * @param nSel nSel
		 * @param ptr ptr
		 */
		static void compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel, void* ptr );

		/** update W
		 *
		 * @param t t
		 * @param ptr ptr
		 * @return something floaty
		 */
		static double update_W(double t, void* ptr );

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
		static void add_new_cut( double *new_col_H, uint32_t *new_cut, uint32_t cut_length, uint32_t nSel, void* ptr );

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
		static void compute_output( double *output, void* ptr );

		/** sort
		 *
		 * @param vals vals
		 * @param idx idx
		 * @param size size
		 */
		static void sort( double* vals, uint32_t* idx, uint32_t size);


	protected:
		/** features */
		CStringFeatures<BYTE>* features;
		/** if bias shall be used */
		bool use_bias;
		/** buffer size */
		INT bufsize;
		/** C1 */
		DREAL C1;
		/** C2 */
		DREAL C2;
		/** epsilon */
		DREAL epsilon;
		/** method */
		E_SVM_TYPE method;

		/** degree */
		INT degree;
		/** from degree */
		INT from_degree;
		/** wd weights */
		SHORTREAL* wd_weights;
		/** num vectors */
		INT num_vec;
		/** length of string in vector */
		INT string_length;
		/** size of alphabet */
		INT alphabet_size;

		/** normalization const */
		DREAL normalization_const;

		/** bias */
		DREAL bias;
		/** w offsets */
		INT* w_offsets;
		/** w dim */
		INT w_dim;
		/** w dim of a single char */
		INT w_dim_single_char;
		/** w */
		SHORTREAL* w;
		/** old w*/
		SHORTREAL* old_w;
		/** nDim big */
		DREAL* tmp_a_buf;
		/** labels */
		DREAL* lab;

		/** cuts */
		SHORTREAL** cuts;
};
#endif
