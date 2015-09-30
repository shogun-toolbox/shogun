/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Adaptation of Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#ifndef _VOWPALWABBIT_H__
#define _VOWPALWABBIT_H__

#include <shogun/lib/config.h>

#include <shogun/classifier/vw/vw_common.h>
#include <shogun/classifier/vw/learners/VwAdaptiveLearner.h>
#include <shogun/classifier/vw/learners/VwNonAdaptiveLearner.h>
#include <shogun/classifier/vw/VwRegressor.h>

#include <shogun/features/streaming/StreamingVwFeatures.h>
#include <shogun/machine/OnlineLinearMachine.h>

namespace shogun
{
/** @brief Class CVowpalWabbit is the implementation of the
 * online learning algorithm used in Vowpal Wabbit.
 *
 * VW is a fast online learning algorithm which operates on
 * sparse features. It uses an online gradient descent technique.
 *
 * For more details, refer to the tutorial at
 * https://github.com/JohnLangford/vowpal_wabbit/wiki/v5.1_tutorial.pdf
 */
class CVowpalWabbit: public COnlineLinearMachine
{
public:

	/** problem type */
	MACHINE_PROBLEM_TYPE(PT_BINARY);

	/**
	 * Default constructor
	 */
	CVowpalWabbit();

	/**
	 * Constructor, taking a features object
	 * as argument
	 *
	 * @param feat StreamingVwFeatures object
	 */
	CVowpalWabbit(CStreamingVwFeatures* feat);

	/** copy constructor
	 * @param vw another VowpalWabbit object
	 */
	CVowpalWabbit(CVowpalWabbit *vw);

	/**
	 * Destructor
	 */
	~CVowpalWabbit();

	/**
	 * Reinitialize the weight vectors.
	 * Call after updating env variables eg. stride.
	 */
	void reinitialize_weights();

	/**
	 * Set whether one desires to not train and only
	 * make passes over all examples instead.
	 *
	 * This is useful if one wants to create a cache file from data.
	 *
	 * @param dont_train true if one doesn't want to train
	 */
	void set_no_training(bool dont_train) { no_training = dont_train; }

	/**
	 * Set whether learning is adaptive or not
	 *
	 * @param adaptive_learning true if adaptive
	 */
	void set_adaptive(bool adaptive_learning);

	/**
	 * Set whether to use the more expensive
	 * exact norm for adaptive learning
	 *
	 * @param exact_adaptive true if exact norm is required
	 */
	void set_exact_adaptive_norm(bool exact_adaptive);

	/**
	 * Set number of passes (only works for cached input)
	 *
	 * @param passes number of passes
	 */
	void set_num_passes(int32_t passes)
	{
		env->num_passes = passes;
	}

	/**
	 * Load regressor from a dump file
	 *
	 * @param file_name name of regressor file
	 */
	void load_regressor(char* file_name);

	/**
	 * Set regressor output parameters
	 *
	 * @param file_name name of file to save regressor to
	 * @param is_text human readable or not, bool
	 */
	void set_regressor_out(char* file_name, bool is_text = true);

	/**
	 * Set file name of prediction output
	 *
	 * @param file_name name of file to save predictions to
	 */
	void set_prediction_out(char* file_name);

	/**
	 * Add a pair of namespaces whose features should
	 * be crossed for quadratic updates
	 *
	 * @param pair a string with the two namespace names concatenated
	 */
	void add_quadratic_pair(char* pair);

	/**
	 * Train on a StreamingVwFeatures object
	 *
	 * @param feat StreamingVwFeatures to train using
	 */
	virtual bool train_machine(CFeatures* feat = NULL);

	/**
	 * Predict for an example
	 *
	 * @param ex VwExample to predict for
	 *
	 * @return prediction
	 */
	virtual float32_t predict_and_finalize(VwExample* ex);

	/**
	 * Computes the exact norm during adaptive learning
	 *
	 * @param ex example
	 * @param sum_abs_x set by reference, sum of abs of features
	 *
	 * @return norm
	 */
	float32_t compute_exact_norm(VwExample* &ex, float32_t& sum_abs_x);

	/**
	 * Computes the exact norm for quadratic features during adaptive learning
	 *
	 * @param weights weights
	 * @param page_feature current feature
	 * @param offer_features paired features
	 * @param mask mask
	 * @param g square of gradient
	 * @param sum_abs_x sum of absolute value of features
	 *
	 * @return norm
	 */
	float32_t compute_exact_norm_quad(float32_t* weights, VwFeature& page_feature, v_array<VwFeature> &offer_features,
					  vw_size_t mask, float32_t g, float32_t& sum_abs_x);

	/**
	 * Get the environment
	 *
	 * @return environment as CVwEnvironment*
	 */
	virtual CVwEnvironment* get_env()
	{
		SG_REF(env);
		return env;
	}

	/**
	 * Return the name of the object
	 *
	 * @return VowpalWabbit
	 */
	virtual const char* get_name() const { return "VowpalWabbit"; }

	/**
	 * Sets the train/update methods depending on parameters
	 * set, eg. adaptive or not
	 */
	virtual void set_learner();

	/**
	 * Get learner
	 */
	CVwLearner* get_learner() { return learner; }

private:
	/**
	 * Initialize members
	 *
	 * @param feat Features object
	 */
	virtual void init(CStreamingVwFeatures* feat = NULL);

	/**
	 * Predict with l1 regularization
	 *
	 * @param ex example
	 *
	 * @return prediction
	 */
	virtual float32_t inline_l1_predict(VwExample* &ex);

	/**
	 * Predict with no regularization term
	 *
	 * @param ex example
	 *
	 * @return prediction
	 */
	virtual float32_t inline_predict(VwExample* &ex);

	/**
	 * Reduce the prediction within limits
	 *
	 * @param ret prediction
	 *
	 * @return prediction within limits
	 */
	virtual float32_t finalize_prediction(float32_t ret);

	/**
	 * Output example, i.e. write prediction, print update etc.
	 *
	 * @param ex example
	 */
	virtual void output_example(VwExample* &ex);

	/**
	 * Print statistics like VW
	 *
	 * @param ex example
	 */
	virtual void print_update(VwExample* &ex);

	/**
	 * Output the prediction to a file
	 *
	 * @param f file descriptor
	 * @param res prediction
	 * @param weight weight of example
	 * @param tag tag
	 */
	virtual void output_prediction(int32_t f, float32_t res, float32_t weight, v_array<char> tag);

	/**
	 * Set whether to display statistics or not
	 *
	 * @param verbose true or false
	 */
	void set_verbose(bool verbose);

protected:
	/// Features
	CStreamingVwFeatures* features;

	/// Environment for VW, i.e., globals
	CVwEnvironment* env;

	/// Learner to use
	CVwLearner* learner;

	/// Regressor
	CVwRegressor* reg;

private:
	/// Whether to display statistics or not
	bool quiet;

	/// Whether we should just run over examples without training
	bool no_training;

	/// Multiplication factor for number of examples to dump after
	float32_t dump_interval;
	/// Sum of loss since last printed update
	float32_t sum_loss_since_last_dump;
	/// Number of weighted examples in previous dump
	float64_t old_weighted_examples;

	/// Name of file to save regressor to
	char* reg_name;
	/// Whether to save regressor as readable text or not
	bool reg_dump_text;

	/// Whether to save predictions or not
	bool save_predictions;
	/// Descriptor of prediction file
	int32_t prediction_fd;
};

}
#endif // _VOWPALWABBIT_H__
