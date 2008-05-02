/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GUICLASSIFIER_H__
#define _GUICLASSIFIER_H__

#include "lib/config.h"
#include "base/SGObject.h"

#ifndef HAVE_SWIG
#include "classifier/Classifier.h"

class CSGInterface;

class CGUIClassifier : public CSGObject
{
	public:
		CGUIClassifier(CSGInterface* interface);
		~CGUIClassifier();

		/** create new classifier */
		bool new_classifier(CHAR* name, INT d=6, INT from_d=40);
		/** set maximum train time */
		bool set_max_train_time(DREAL max);
		/** test classifier */
		bool test(CHAR* filename_out=NULL, CHAR* filename_roc=NULL);
		/** load classifier from file */
		bool load(CHAR* filename, CHAR* type);
		bool save(CHAR* param);
		CLabels* classify(CLabels* output=NULL);
		CLabels* classify_kernelmachine(CLabels* output=NULL);
		CLabels* classify_distancemachine(CLabels* output=NULL);
		CLabels* classify_linear(CLabels* output=NULL);
		CLabels* classify_sparse_linear(CLabels* output=NULL);
		CLabels* classify_byte_linear(CLabels* output=NULL);
		bool classify_example(INT idx, DREAL& result);
		inline CClassifier* get_classifier() { return classifier; }

		bool get_trained_classifier(DREAL* &weights, INT& rows,
				INT& cols, DREAL*& bias, INT& brows, INT& bcols);
		bool get_svm(DREAL* &weights, INT& rows, INT& cols,
				DREAL*& bias, INT& brows, INT& bcols);
		bool get_linear(DREAL* &weights, INT& rows, INT& cols,
				DREAL*& bias, INT& brows, INT& bcols);
		bool get_sparse_linear(DREAL* &weights, INT& rows, INT& cols,
				DREAL*& bias, INT& brows, INT& bcols);
		bool get_clustering(DREAL* &weights, INT& rows, INT& cols,
				DREAL*& bias, INT& brows, INT& bcols);

		/// perceptron learnrate & maxiter
		bool set_perceptron_parameters(DREAL lernrate, INT maxiter);

		/// SVM functions
		bool set_svm_C(DREAL C1, DREAL C2);
		bool set_svm_bufsize(INT bufsize);
		bool set_svm_qpsize(INT qpsize);
		bool set_svm_max_qpsize(INT max_qpsize);
		bool set_svm_mkl_enabled(bool enabled);
		bool set_svm_shrinking_enabled(bool enabled);
		bool set_svm_one_class_nu(DREAL nu);
		bool set_svm_batch_computation_enabled(bool enabled);
		bool set_do_auc_maximization(bool do_auc);
		bool set_svm_linadd_enabled(bool enabled);
		bool set_svm_bias_enabled(bool enabled);
		bool set_svm_epsilon(DREAL epsilon);
		bool set_svr_tube_epsilon(DREAL tube_epsilon);
		bool set_svm_mkl_parameters(DREAL weight_epsilon, DREAL C_mkl);
		bool set_svm_precompute_enabled(INT precompute);

		/** train SVM */
		bool train_svm();
		/** train K-nearest-neighbour */
		bool train_knn(INT k=3);
		/** train clustering */
		bool train_clustering(INT k=3, INT max_iter=1000);
		/** train linear classifier */
		bool train_linear();
		/** train sparse linear classifier */
		bool train_sparse_linear();
		/** train WD OCAS */
		bool train_wdocas();

	protected:
		CSGInterface* ui;
		CClassifier* classifier;
		double max_train_time;

		double perceptron_learnrate;
		int perceptron_maxiter;

		int svm_qpsize;
		int svm_bufsize;
		int svm_max_qpsize;
		double svm_weight_epsilon;
		double svm_epsilon;
		double svm_tube_epsilon;
		double svm_nu;
		double svm_C1;
		double svm_C2;
		double svm_C_mkl;
		bool svm_use_bias;
		bool svm_use_mkl;
		bool svm_use_batch_computation;
		bool svm_use_linadd;
		bool svm_use_precompute;
		bool svm_use_precompute_subkernel;
		bool svm_use_precompute_subkernel_light;
		bool svm_use_shrinking;
		bool svm_do_auc_maximization;
};
#endif //HAVE_SWIG
#endif
