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

#include <shogun/classifier/vw/vw_common.h>
#include <shogun/classifier/vw/learners/VwAdaptiveLearner.h>
#include <shogun/classifier/vw/learners/VwNonAdaptiveLearner.h>
#include <shogun/classifier/vw/VwRegressor.h>

#include <shogun/features/StreamingVwFeatures.h>
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
					  size_t mask, float32_t g, float32_t& sum_abs_x);

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

private:
	/**
	 * Initialize members
	 *
	 * @param feat Features object
	 */
	virtual void init(CStreamingVwFeatures* feat = NULL);

	/**
	 * Sets the train/update methods depending on parameters
	 * set, eg. adaptive or not
	 */
	virtual void set_learner();

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
	 * Print statistics like VW
	 *
	 * @param ex example
	 */
	virtual void print_update(VwExample* &ex);

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

	/// Name of file to save regressor to
	char* reg_name;

	/// Whether to save regressor as readable text or not
	bool reg_dump_text;
};

}
#endif // _VOWPALWABBIT_H__
