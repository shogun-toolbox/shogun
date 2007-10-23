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

#ifndef _SVMOCAS_H___
#define _SVMOCAS_H___

#include "lib/common.h"
#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

enum E_SVM_TYPE
{
	SVM_OCAS = 0,
	SVM_BMRM = 1
};

class CSVMOcas : public CSparseLinearClassifier
{
	public:
		CSVMOcas(E_SVM_TYPE);
		CSVMOcas(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab);
		virtual ~CSVMOcas();

		virtual inline EClassifierType get_classifier_type() { return CT_SVMOCAS; }
		virtual bool train();

		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		inline DREAL get_C1() { return C1; }
		inline DREAL get_C2() { return C2; }

		inline void set_epsilon(DREAL eps) { epsilon=eps; }
		inline DREAL get_epsilon() { return epsilon; }

		void mul_sparse_col(double alpha, CSparseFeatures<DREAL>* sparse_mat, uint32_t col);
		void add_sparse_col(double *full_vec, CSparseFeatures<DREAL>* sparse_mat, uint32_t col);
		double dp_sparse_col(double *full_vec, CSparseFeatures<DREAL>* sparse_mat, uint32_t col);
		static double sparse_update_W(double t );
		static void sparse_add_new_cut( double *new_col_H, uint32_t *new_cut, uint32_t cut_length, uint32_t nSel );
		static void sparse_compute_output( double *output );
		static void sparse_compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel );

	protected:
		DREAL C1;
		DREAL C2;
		DREAL epsilon;
		E_SVM_TYPE method;
};
#endif
