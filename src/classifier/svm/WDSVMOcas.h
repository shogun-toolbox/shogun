/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Vojtech Franc 
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WDSVMOCAS_H___
#define _WDSVMOCAS_H___

#include "lib/common.h"
#include "classifier/Classifier.h"
#include "classifier/svm/SVMOcas.h"
#include "features/StringFeatures.h"
#include "features/Labels.h"

class CWDSVMOcas : public CClassifier
{
	public:
		CWDSVMOcas(E_SVM_TYPE);
		CWDSVMOcas(DREAL C, CStringFeatures<BYTE>* traindat, CLabels* trainlab);
		virtual ~CWDSVMOcas();

		virtual inline EClassifierType get_classifier_type() { return CT_WDSVMOCAS; }
		virtual bool train();

		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		inline DREAL get_C1() { return C1; }
		inline DREAL get_C2() { return C2; }

		inline void set_epsilon(DREAL eps) { epsilon=eps; }
		inline DREAL get_epsilon() { return epsilon; }

		inline void set_features(CStringFeatures<BYTE>* feat) { features=feat; }
		inline CStringFeatures<BYTE>* get_features() { return features; }

		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }
		inline bool get_bias_enabled() { return use_bias; }

		inline void set_bufsize(INT sz) { bufsize=sz; }
		inline INT get_bufsize() { return bufsize; }

	protected:

		static void compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel, void* ptr );
		static double update_W(double t, void* ptr );
		static void add_new_cut( double *new_col_H, uint32_t *new_cut, uint32_t cut_length, uint32_t nSel, void* ptr );
		static void compute_output( double *output, void* ptr );
		static void sort( double* vals, uint32_t* idx, uint32_t size);


	protected:
		CStringFeatures<BYTE>* features;
		bool use_bias;
		INT bufsize;
		DREAL C1;
		DREAL C2;
		DREAL epsilon;
		E_SVM_TYPE method;

		INT degree;
		DREAL* wd_weights;
		INT string_length;
		INT alphabet_size;

		DREAL bias;
		INT w_dim_single_char;
		INT w_dim;
		DREAL* w;
		DREAL* old_w;
		DREAL* tmp_a_buf; /// nDim big
		DREAL* lab;

		DREAL** cuts;
};
#endif
