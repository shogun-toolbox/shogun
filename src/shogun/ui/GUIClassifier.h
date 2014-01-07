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

#include <lib/config.h>
#include <base/SGObject.h>
#include <machine/Machine.h>
#include <classifier/svm/SVM.h>

namespace shogun
{
class CSGInterface;

/** @brief UI classifier */
class CGUIClassifier : public CSGObject
{
	public:
		/** constructor */
		CGUIClassifier() { };
		/** constructor
		 * @param interface
		 */
		CGUIClassifier(CSGInterface* interface);
		/** destructor */
		~CGUIClassifier();

		/** create new classifier */
		bool new_classifier(char* name, int32_t d=6, int32_t from_d=40);
		/** set maximum train time */
		bool set_max_train_time(float64_t max);
		/** load classifier from file */
		bool load(char* filename, char* type);
		/** save
		 * @param param
		 */
		bool save(char* param);
		/** classify */
		CLabels* classify();
		/** classify kernel machine */
		CLabels* classify_kernelmachine();
		/** classify distance machine */
		CLabels* classify_distancemachine();
		/** classify linear */
		CLabels* classify_linear();
		/** classify byte linear */
		CLabels* classify_byte_linear();
		/** classify example
		 * @param idx
		 * @param result
		 */
		bool classify_example(int32_t idx, float64_t& result);
		/** get classifier */
		inline CMachine* get_classifier() { return classifier; }

		/** get trained classifier
		 * @param weights
		 * @param rows
		 * @param cols
		 * @param bias
		 * @param brows
		 * @param bcols
		 * @param idx
		 */
		bool get_trained_classifier(
			float64_t* &weights, int32_t& rows, int32_t& cols,
			float64_t*& bias, int32_t& brows, int32_t& bcols,
			int32_t idx=-1); // which SVM in Multiclass

		/** get number of SVMs in Multiclass */
		int32_t get_num_svms();
		/** get svm
		 * @param weights
		 * @param rows
		 * @param cols
		 * @param bias
		 * @param brows
		 * @param bcols
		 * @param idx
		 */
		bool get_svm(
			float64_t* &weights, int32_t& rows, int32_t& cols,
			float64_t*& bias, int32_t& brows, int32_t& bcols,
			int32_t idx=-1); // which SVM in Multiclass
		/** get linear
		 * @param weights
		 * @param rows
		 * @param cols
		 * @param bias
		 * @param brows
		 * @param bcols
		 */
		bool get_linear(
			float64_t* &weights, int32_t& rows, int32_t& cols,
			float64_t*& bias, int32_t& brows, int32_t& bcols);
		/** get clustering
		 * @param weights
		 * @param rows
		 * @param cols
		 * @param bias
		 * @param brows
		 * @param bcols
		 */
		bool get_clustering(
			float64_t* &weights, int32_t& rows, int32_t& cols,
			float64_t*& bias, int32_t& brows, int32_t& bcols);

		// perceptron learnrate & maxiter
		/** set perceptron parameters
		 * @param lernrate
		 * @param maxiter
		 */
		bool set_perceptron_parameters(float64_t lernrate, int32_t maxiter);

		// SVM functions
		/** set svm C
		 * @param C1
		 * @param C2
		 */
		bool set_svm_C(float64_t C1, float64_t C2);
		/** set svm bufsize
		 * @param bufsize
		 */
		bool set_svm_bufsize(int32_t bufsize);
		/** set svm qpsize
		 * @param qpsize
		 */
		bool set_svm_qpsize(int32_t qpsize);
		/** set svm max qpsize
		 * @param max_qpsize
		 */
		bool set_svm_max_qpsize(int32_t max_qpsize);
		/** set svm shrinking enabled
		 * @param enabled
		 */
		bool set_svm_shrinking_enabled(bool enabled);
		/** set svm nu
		 * @param nu
		 */
		bool set_svm_nu(float64_t nu);
		/** set svm batch computation enabled
		 * @param enabled
		 */
		bool set_svm_batch_computation_enabled(bool enabled);
		/** set do auc maximization
		 * @param do_auc
		 */
		bool set_do_auc_maximization(bool do_auc);
		/** set svm linadd enabled
		 * @param enabled
		 */
		bool set_svm_linadd_enabled(bool enabled);
		/** set svm bias enabled
		 * @param enabled
		 */
		bool set_svm_bias_enabled(bool enabled);
		/** set mkl interleaved enabled
		 * @param enabled
		 */
		bool set_mkl_interleaved_enabled(bool enabled);
		/** set svm epsilon
		 * @param epsilon
		 */
		bool set_svm_epsilon(float64_t epsilon);
		/** set svr tube epsilon
		 * @param tube_epsilon
		 */
		bool set_svr_tube_epsilon(float64_t tube_epsilon);
		/** set svm mkl parameters
		 * @param weight_epsilon
		 * @param C_mkl
		 * @param mkl_norm
		 */
		bool set_svm_mkl_parameters(
			float64_t weight_epsilon, float64_t C_mkl, float64_t mkl_norm);
		/** set mkl block norm
		 * @param mkl_bnorm
		 */
		bool set_mkl_block_norm(float64_t mkl_bnorm);
		/** set elasticnet lambda
		 * @param lambda
		 */
		bool set_elasticnet_lambda(float64_t lambda);
		/** set svm precompute enabled
		 * @param precompute
		 */
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
		virtual const char* get_name() const { return "GUIClassifier"; }

	protected:
		/** ui */
		CSGInterface* ui;
		/** classifier */
		CMachine* classifier;
		/** max train time */
		float64_t max_train_time;
		/** perceptron learnrate */
		float64_t perceptron_learnrate;
		/** perceptron maxiter */
		int32_t perceptron_maxiter;
		/** svm qpsize */
		int32_t svm_qpsize;
		/** svm bufsize */
		int32_t svm_bufsize;
		/** svm max qpsize */
		int32_t svm_max_qpsize;
		/** mkl norm */
		float64_t mkl_norm;
		/** mkl block norm */
		float64_t mkl_block_norm;
		/** ent lambda */
		float64_t ent_lambda;
		/** svm weight epsilon */
		float64_t svm_weight_epsilon;
		/** svm epsilon */
		float64_t svm_epsilon;
		/** svm tube epsilon */
		float64_t svm_tube_epsilon;
		/** svm nu */
		float64_t svm_nu;
		/** svm C1 */
		float64_t svm_C1;
		/** svm C2 */
		float64_t svm_C2;
		/** C mkl */
		float64_t C_mkl;
		/** krr tau */
		float64_t krr_tau;
		/** mkl use interleaved */
		bool mkl_use_interleaved;
		/** svm use bias */
		bool svm_use_bias;
		/** svm use batch computation */
		bool svm_use_batch_computation;
		/** svm use linadd */
		bool svm_use_linadd;
		/** svm use precompute */
		bool svm_use_precompute;
		/** svm use precompute subkernel */
		bool svm_use_precompute_subkernel;
		/** svm use precompute subkernel light */
		bool svm_use_precompute_subkernel_light;
		/** svm use shrinking */
		bool svm_use_shrinking;
		/** svm do auc maximization */
		bool svm_do_auc_maximization;

		/** constraint generator */
		CSVM* constraint_generator;
		/** solver type */
		ESolverType solver_type;
};
}
#endif
