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

#include <shogun/classifier/vw/VowpalWabbit.h>

using namespace shogun;

CVowpalWabbit::CVowpalWabbit()
	: COnlineLinearMachine()
{
	reg=NULL;
	learner=NULL;
	init(NULL);
}

CVowpalWabbit::CVowpalWabbit(CStreamingVwFeatures* feat)
	: COnlineLinearMachine()
{
	reg=NULL;
	learner=NULL;
	init(feat);
}

CVowpalWabbit::~CVowpalWabbit()
{
	SG_UNREF(env);
	SG_UNREF(reg);
	SG_UNREF(learner);
}

void CVowpalWabbit::set_adaptive(bool adaptive_learning)
{
	if (adaptive_learning)
	{
		env->adaptive = true;
		env->stride = 2;
		env->power_t = 0.;
	}
	else
		env->adaptive = false;
}

void CVowpalWabbit::set_regressor_out(char* file_name, bool is_text)
{
	reg_name = file_name;
	reg_dump_text = is_text;
}

void CVowpalWabbit::add_quadratic_pair(char* pair)
{
	env->pairs.push_back(pair);
}

bool CVowpalWabbit::train_machine(CStreamingVwFeatures* feat)
{
	ASSERT(features);

	set_learner();

	VwExample* example = NULL;
	size_t current_pass = 0;
	float32_t dump_interval = exp(1.);

	const char* header_fmt = "%-10s %-10s %8s %8s %10s %8s %8s\n";

	if (!quiet)
	{
		SG_SPRINT(header_fmt,
			  "average", "since", "example", "example",
			  "current", "current", "current");
		SG_SPRINT(header_fmt,
			  "loss", "last", "counter", "weight", "label", "predict", "features");
	}

	features->start_parser();
	while (env->passes_complete < env->num_passes)
	{
		while (features->get_next_example())
		{
			example = features->get_example();

			if (example->pass != current_pass)
			{
				env->eta *= env->eta_decay_rate;
				current_pass = example->pass;
			}

			predict_and_finalize(example);

			learner->train(example, example->eta_round);
			example->eta_round = 0.;

			if (!quiet)
			{
				if (env->weighted_examples + example->ld->weight > dump_interval)
				{
					print_update(example);
					dump_interval *= 2;
				}
			}

			features->release_example();
		}
		env->passes_complete++;
		if (env->passes_complete < env->num_passes)
			features->reset_stream();
	}
	features->end_parser();

	if (env->l1_regularization > 0.)
	{
		uint32_t length = 1 << env->num_bits;
		size_t stride = env->stride;
		float32_t gravity = env->l1_regularization * env->update_sum;
		for (uint32_t i = 0; i < length; i++)
			reg->weight_vectors[0][stride*i] = real_weight(reg->weight_vectors[0][stride*i], gravity);
	}

	if (reg_name != NULL)
		reg->dump_regressor(reg_name, reg_dump_text);

	return true;
}

float32_t CVowpalWabbit::predict_and_finalize(VwExample* ex)
{
	float32_t prediction;
	if (env->l1_regularization != 0.)
		prediction = inline_l1_predict(ex);
	else
		prediction = inline_predict(ex);

	ex->final_prediction = 0;
	ex->final_prediction += prediction;
	ex->final_prediction = finalize_prediction(ex->final_prediction);
	float32_t t = ex->example_t;

	if (ex->ld->label != FLT_MAX)
	{
		ex->loss = reg->get_loss(ex->final_prediction, ex->ld->label) * ex->ld->weight;
		float64_t update = 0.;
		update = (env->eta)/pow(t, env->power_t) * ex->ld->weight;
		ex->eta_round = reg->get_update(ex->final_prediction, ex->ld->label, update, ex->total_sum_feat_sq);
		env->update_sum += update;
	}

	return prediction;
}

void CVowpalWabbit::init(CStreamingVwFeatures* feat)
{
	features = feat;
	env = feat->get_env();
	reg = new CVwRegressor(env);
	SG_REF(env);
	SG_REF(reg);

	quiet = false;
	reg_name = NULL;
	reg_dump_text = true;

	w = reg->weight_vectors[0];
	w_dim = 1 << env->num_bits;
	bias = 0.;
}

void CVowpalWabbit::set_learner()
{
	if (env->adaptive)
		learner = new CVwAdaptiveLearner(reg, env);
	else
		learner = new CVwNonAdaptiveLearner(reg, env);
	SG_REF(learner);
}

float32_t CVowpalWabbit::inline_l1_predict(VwExample* &ex)
{
	size_t thread_num = 0;

	float32_t prediction = ex->ld->get_initial();

	float32_t* weights = reg->weight_vectors[thread_num];
	size_t thread_mask = env->thread_mask;

	prediction += features->dense_dot_truncated(weights, ex, env->l1_regularization * env->update_sum);

	for (int32_t k = 0; k < env->pairs.get_num_elements(); k++)
	{
		char* i = env->pairs.get_element(k);

		v_array<VwFeature> temp = ex->atomics[(int32_t)(i[0])];
		temp.begin = ex->atomics[(int32_t)(i[0])].begin;
		temp.end = ex->atomics[(int32_t)(i[0])].end;
		for (; temp.begin != temp.end; temp.begin++)
			prediction += one_pf_quad_predict_trunc(weights, *temp.begin,
								ex->atomics[(int32_t)(i[1])], thread_mask,
								env->l1_regularization * env->update_sum);
	}

	return prediction;
}

float32_t CVowpalWabbit::inline_predict(VwExample* &ex)
{
	size_t thread_num = 0;
	float32_t prediction = ex->ld->initial;

	float32_t* weights = reg->weight_vectors[thread_num];
	size_t thread_mask = env->thread_mask;
	prediction += features->dense_dot(weights, 0);

	for (int32_t k = 0; k < env->pairs.get_num_elements(); k++)
	{
		char* i = env->pairs.get_element(k);

		v_array<VwFeature> temp = ex->atomics[(int32_t)(i[0])];
		temp.begin = ex->atomics[(int32_t)(i[0])].begin;
		temp.end = ex->atomics[(int32_t)(i[0])].end;
		for (; temp.begin != temp.end; temp.begin++)
			prediction += one_pf_quad_predict(weights, *temp.begin,
							  ex->atomics[(int32_t)(i[1])],
							  thread_mask);
	}

	return prediction;
}

float32_t CVowpalWabbit::finalize_prediction(float32_t ret)
{
	if (isnan(ret))
		return 0.5;
	if (ret > env->max_label)
		return env->max_label;
	if (ret < env->min_label)
		return env->min_label;

	return ret;
}

void CVowpalWabbit::print_update(VwExample* &ex)
{
	SG_SPRINT("%-10.6f %-10.6f %8lld %8.1f %8.4f %8.4f %8lu\n",
		  env->sum_loss/env->weighted_examples,
		  0.0,
		  env->example_number,
		  env->weighted_examples,
		  ex->ld->label,
		  ex->final_prediction,
		  (long unsigned int)ex->num_features);
}
