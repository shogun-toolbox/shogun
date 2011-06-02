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

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/machine/Machine.h>
#include <shogun/classifier/svm/SVM.h>

namespace shogun
{
class CSGInterface;

class CGUIClassifier : public CSGObject
{
	public:
		CGUIClassifier(CSGInterface* interface);
		~CGUIClassifier();

		/** create new classifier */
		bool new_classifier(char* name, int32_t d=6, int32_t from_d=40);
		/** set maximum train time */
		bool set_max_train_time(float64_t max);
		/** load classifier from file */
		bool load(char* filename, char* type);
		bool save(char* param);
		CLabels* classify();
		CLabels* classify_kernelmachine();
		CLabels* classify_distancemachine();
		CLabels* classify_linear();
		CLabels* classify_byte_linear();
		bool classify_example(int32_t idx, float64_t& result);
		inline CMachine* get_classifier() { return classifier; }

		bool get_trained_classifier(
			float64_t* &weights, int32_t& rows, int32_t& cols,
			float64_t*& bias, int32_t& brows, int32_t& bcols,
			int32_t idx=-1); // which SVM in MultiClass

		/** get number of SVMs in MultiClass */
		int32_t get_num_svms();
		bool get_svm(
			float64_t* &weights, int32_t& rows, int32_t& cols,
			float64_t*& bias, int32_t& brows, int32_t& bcols,
			int32_t idx=-1); // which SVM in MultiClass

		bool get_linear(
			float64_t* &weights, int32_t& rows, int32_t& cols,
			float64_t*& bias, int32_t& brows, int32_t& bcols);

		bool get_clustering(
			float64_t* &weights, int32_t& rows, int32_t& cols,
			float64_t*& bias, int32_t& brows, int32_t& bcols);

		/// perceptron learnrate & maxiter
		bool set_perceptron_parameters(float64_t lernrate, int32_t maxiter);

		/// SVM functions
		bool set_svm_C(float64_t C1, float64_t C2);
		bool set_svm_bufsize(int32_t bufsize);
		bool set_svm_qpsize(int32_t qpsize);
		bool set_svm_max_qpsize(int32_t max_qpsize);
		bool set_svm_shrinking_enabled(bool enabled);
		bool set_svm_nu(float64_t nu);
		bool set_svm_batch_computation_enabled(bool enabled);
		bool set_do_auc_maximization(bool do_auc);
		bool set_svm_linadd_enabled(bool enabled);
		bool set_svm_bias_enabled(bool enabled);
		bool set_mkl_interleaved_enabled(bool enabled);
		bool set_svm_epsilon(float64_t epsilon);
		bool set_svr_tube_epsilon(float64_t tube_epsilon);
		bool set_svm_mkl_parameters(
			float64_t weight_epsilon, float64_t C_mkl, float64_t mkl_norm);
		bool set_mkl_block_norm(float64_t mkl_bnorm);
		bool set_elasticnet_lambda(float64_t lambda);
		bool set_svm_precompute_enabled(int32_t precompute);

		/** set KRR's tau */
		bool set_krr_tau(float64_t tau=1);
		/** set solver type */
		bool set_solver(char* solver);
		/** set constraint generator */
		bool set_constraint_generator(char* cg);

		/** train MKL multiclass*/
		bool train_mkl_multiclass();
		/** train MKL */
		bool train_mkl();
		/** train SVM */
		bool train_svm();
		/** train K-nearest-neighbour */
		bool train_knn(int32_t k=3);
		/** train kernel ridge regression */
		bool train_krr();
		/** train clustering */
		bool train_clustering(int32_t k=3, int32_t max_iter=1000);
		/** train linear classifier
		 * @param gamma gamma parameter of LDA
		 */
		bool train_linear(float64_t gamma=0);
		/** train sparse linear classifier */
		bool train_sparse_linear();
		/** train WD OCAS */
		bool train_wdocas();

		/** @return object name */
		inline virtual const char* get_name() const { return "GUIClassifier"; }

	protected:
		CSGInterface* ui;
		CMachine* classifier;
		float64_t max_train_time;

		float64_t perceptron_learnrate;
		int32_t perceptron_maxiter;

		int32_t svm_qpsize;
		int32_t svm_bufsize;
		int32_t svm_max_qpsize;
		float64_t mkl_norm;
		float64_t mkl_block_norm;
		float64_t ent_lambda;
		float64_t svm_weight_epsilon;
		float64_t svm_epsilon;
		float64_t svm_tube_epsilon;
		float64_t svm_nu;
		float64_t svm_C1;
		float64_t svm_C2;
		float64_t C_mkl;
		float64_t krr_tau;
		bool mkl_use_interleaved;
		bool svm_use_bias;
		bool svm_use_batch_computation;
		bool svm_use_linadd;
		bool svm_use_precompute;
		bool svm_use_precompute_subkernel;
		bool svm_use_precompute_subkernel_light;
		bool svm_use_shrinking;
		bool svm_do_auc_maximization;

		CSVM* constraint_generator;
		ESolverType solver_type;
};
}
#endif
