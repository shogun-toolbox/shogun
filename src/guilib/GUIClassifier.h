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
		bool new_classifier(char* name, int32_t d=6, int32_t from_d=40);
		/** set maximum train time */
		bool set_max_train_time(DREAL max);
		/** test classifier */
		bool test(char* filename_out=NULL, char* filename_roc=NULL);
		/** load classifier from file */
		bool load(char* filename, char* type);
		bool save(char* param);
		CLabels* classify(CLabels* output=NULL);
		CLabels* classify_kernelmachine(CLabels* output=NULL);
		CLabels* classify_distancemachine(CLabels* output=NULL);
		CLabels* classify_linear(CLabels* output=NULL);
		CLabels* classify_sparse_linear(CLabels* output=NULL);
		CLabels* classify_byte_linear(CLabels* output=NULL);
		bool classify_example(int32_t idx, DREAL& result);
		inline CClassifier* get_classifier() { return classifier; }

		bool get_trained_classifier(DREAL* &weights, int32_t& rows,
				int32_t& cols, DREAL*& bias, int32_t& brows, int32_t& bcols);
		bool get_svm(DREAL* &weights, int32_t& rows, int32_t& cols,
				DREAL*& bias, int32_t& brows, int32_t& bcols);
		bool get_linear(DREAL* &weights, int32_t& rows, int32_t& cols,
				DREAL*& bias, int32_t& brows, int32_t& bcols);
		bool get_sparse_linear(DREAL* &weights, int32_t& rows, int32_t& cols,
				DREAL*& bias, int32_t& brows, int32_t& bcols);
		bool get_clustering(DREAL* &weights, int32_t& rows, int32_t& cols,
				DREAL*& bias, int32_t& brows, int32_t& bcols);

		/// perceptron learnrate & maxiter
		bool set_perceptron_parameters(DREAL lernrate, int32_t maxiter);

		/// SVM functions
		bool set_svm_C(DREAL C1, DREAL C2);
		bool set_svm_bufsize(int32_t bufsize);
		bool set_svm_qpsize(int32_t qpsize);
		bool set_svm_max_qpsize(int32_t max_qpsize);
		bool set_svm_mkl_enabled(bool enabled);
		bool set_svm_shrinking_enabled(bool enabled);
		bool set_svm_one_class_nu(DREAL nu);
		bool set_svm_batch_computation_enabled(bool enabled);
		bool set_do_auc_maximization(bool do_auc);
		bool set_svm_linadd_enabled(bool enabled);
		bool set_svm_bias_enabled(bool enabled);
		bool set_svm_epsilon(DREAL epsilon);
		bool set_svr_tube_epsilon(DREAL tube_epsilon);
		bool set_svm_mkl_parameters(DREAL weight_epsilon, DREAL C_mkl, int32_t mkl_norm);
		bool set_svm_precompute_enabled(int32_t precompute);

		/** set KRR's tau */
		bool set_krr_tau(DREAL tau=1);

		/** train SVM */
		bool train_svm();
		/** train K-nearest-neighbour */
		bool train_knn(int32_t k=3);
		/** train clustering */
		bool train_clustering(int32_t k=3, int32_t max_iter=1000);
		/** train linear classifier
		 * @param gamma gamma parameter of LDA
		 */
		bool train_linear(DREAL gamma=0);
		/** train sparse linear classifier */
		bool train_sparse_linear();
		/** train WD OCAS */
		bool train_wdocas();

	protected:
		CSGInterface* ui;
		CClassifier* classifier;
		double max_train_time;

		double perceptron_learnrate;
		int32_t perceptron_maxiter;

		int32_t svm_qpsize;
		int32_t svm_bufsize;
		int32_t svm_max_qpsize;
		int32_t svm_mkl_norm;
		double svm_weight_epsilon;
		double svm_epsilon;
		double svm_tube_epsilon;
		double svm_nu;
		double svm_C1;
		double svm_C2;
		double svm_C_mkl;
		double krr_tau;
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
