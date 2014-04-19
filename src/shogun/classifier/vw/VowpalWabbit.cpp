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

#include <algorithm>
#include <shogun/classifier/vw/VowpalWabbit.h>
#include <fcntl.h>

using namespace std;
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

CVowpalWabbit::CVowpalWabbit(CVowpalWabbit *vw)
	: COnlineLinearMachine()
{
	features = vw->features;
	env = vw->env;
	reg = new CVwRegressor(env);
	SG_REF(env);
	SG_REF(reg);

	quiet = vw->quiet;
	no_training = vw->no_training;
	dump_interval = vw->dump_interval;
	sum_loss_since_last_dump = 0.;
	reg_name = vw->reg_name;
	reg_dump_text = vw->reg_dump_text;
	save_predictions = vw->save_predictions;
	prediction_fd = vw->prediction_fd;

	w = reg->weight_vectors[0];
	reg->weight_vectors[0] = NULL;
	copy(vw->w, vw->w+vw->w_dim, w);
	w_dim = vw->w_dim;
	bias = vw->bias;
}

CVowpalWabbit::~CVowpalWabbit()
{
	SG_UNREF(env);
	SG_UNREF(reg);
	SG_UNREF(learner);

	// Need to unset w, because we're allocating w as well.  Otherwise
	// OnlineLinearMachine may try to free it as well: (double-free)...
	// w = NULL;
}

void CVowpalWabbit::reinitialize_weights()
{
	if (reg->weight_vectors)
	{
		if (reg->weight_vectors[0])
			SG_FREE(reg->weight_vectors[0]);
		SG_FREE(reg->weight_vectors);
	}

	reg->init(env);
	w = reg->weight_vectors[0];
	reg->weight_vectors[0] = NULL;
}

void CVowpalWabbit::set_adaptive(bool adaptive_learning)
{
	if (adaptive_learning)
	{
		env->adaptive = true;
		env->set_stride(2);
		env->power_t = 0.;
		reinitialize_weights();
	}
	else
		env->adaptive = false;
}

void CVowpalWabbit::set_exact_adaptive_norm(bool exact_adaptive)
{
	if (exact_adaptive)
	{
		set_adaptive(true);
		env->exact_adaptive_norm = true;
	}
	else
		env->exact_adaptive_norm = false;
}

void CVowpalWabbit::load_regressor(char* file_name)
{
	reg->load_regressor(file_name);
	w = reg->weight_vectors[0];
	reg->weight_vectors[0] = NULL;
	w_dim = 1 << env->num_bits;
}

void CVowpalWabbit::set_regressor_out(char* file_name, bool is_text)
{
	reg_name = file_name;
	reg_dump_text = is_text;
}

void CVowpalWabbit::set_prediction_out(char* file_name)
{
	save_predictions = true;
	prediction_fd = open(file_name, O_CREAT|O_TRUNC|O_WRONLY, 0666);
	if (prediction_fd < 0)
		SG_SERROR("Unable to open prediction file %s for writing!\n", file_name)
}

void CVowpalWabbit::add_quadratic_pair(char* pair)
{
	env->pairs.push_back(pair);
}

bool CVowpalWabbit::train_machine(CFeatures* feat)
{
	ASSERT(features || feat)
	if (feat && (features != (CStreamingVwFeatures*) feat))
	{
		SG_UNREF(features);
		init((CStreamingVwFeatures*) feat);
	}

	set_learner();

	VwExample* example = NULL;
	vw_size_t current_pass = 0;

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

			// Check if we shouldn't train (generally used for cache creation)
			if (!no_training)
			{
				if (example->pass != current_pass)
				{
					env->eta *= env->eta_decay_rate;
					current_pass = example->pass;
				}

				predict_and_finalize(example);

				learner->train(example, example->eta_round);
				example->eta_round = 0.;

				output_example(example);
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
		vw_size_t stride = env->stride;
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
		if (env->adaptive && env->exact_adaptive_norm)
		{
			float32_t sum_abs_x = 0.;
			float32_t exact_norm = compute_exact_norm(ex, sum_abs_x);
			update = (env->eta * exact_norm)/sum_abs_x;
			env->update_sum += update;
			ex->eta_round = reg->get_update(ex->final_prediction, ex->ld->label, update, exact_norm);
		}
		else
		{
			update = (env->eta)/pow(t, env->power_t) * ex->ld->weight;
			ex->eta_round = reg->get_update(ex->final_prediction, ex->ld->label, update, ex->total_sum_feat_sq);
		}
		env->update_sum += update;
	}

	return prediction;
}

void CVowpalWabbit::init(CStreamingVwFeatures* feat)
{
	features = feat;

	if (feat)
		env = feat->get_env();
	else
	{
		env=new CVwEnvironment();
		SG_REF(env);
	}

	reg = new CVwRegressor(env);
	SG_REF(reg);

	quiet = true;
	no_training = false;
	dump_interval = exp(1.);
	sum_loss_since_last_dump = 0.;
	reg_name = NULL;
	reg_dump_text = true;
	save_predictions = false;
	prediction_fd = -1;

	w = reg->weight_vectors[0];
	reg->weight_vectors[0] = NULL;
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
	vw_size_t thread_num = 0;

	float32_t prediction = ex->ld->get_initial();

	float32_t* weights = reg->weight_vectors[thread_num];
	vw_size_t thread_mask = env->thread_mask;

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
	vw_size_t thread_num = 0;
	float32_t prediction = ex->ld->initial;

	float32_t* weights = reg->weight_vectors[thread_num];
	vw_size_t thread_mask = env->thread_mask;
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

void CVowpalWabbit::output_example(VwExample* &example)
{
	if (!quiet)
	{
		sum_loss_since_last_dump += example->loss;
		if (env->weighted_examples + example->ld->weight > dump_interval)
		{
			print_update(example);
			dump_interval *= 2;
		}
	}

	if (save_predictions)
	{
		float32_t wt = 0.;
		if (reg->weight_vectors)
			wt = reg->weight_vectors[0][0];

		output_prediction(prediction_fd, example->final_prediction, wt * example->global_weight, example->tag);
	}
}

void CVowpalWabbit::print_update(VwExample* &ex)
{
	SG_SPRINT("%-10.6f %-10.6f %8lld %8.1f   %8.4f %8.4f %8lu\n",
		  (env->sum_loss + ex->loss)/(env->weighted_examples + ex->ld->weight),
		  sum_loss_since_last_dump/(env->weighted_examples + ex->ld->weight - old_weighted_examples),
		  env->example_number + 1,
		  env->weighted_examples + ex->ld->weight,
		  ex->ld->label,
		  ex->final_prediction,
		  (long unsigned int)ex->num_features);
	sum_loss_since_last_dump = 0.0;
	old_weighted_examples = env->weighted_examples + ex->ld->weight;
}


void CVowpalWabbit::output_prediction(int32_t f, float32_t res, float32_t weight, v_array<char> tag)
{
	if (f >= 0)
	{
		char temp[30];
		int32_t num = sprintf(temp, "%f", res);
		ssize_t t;
		t = write(f, temp, num);
		if (t != num)
			SG_SERROR("Write error!\n")

		if (tag.begin != tag.end)
		{
			temp[0] = ' ';
			t = write(f, temp, 1);
			if (t != 1)
				SG_SERROR("Write error!\n")

			t = write(f, tag.begin, sizeof(char)*tag.index());
			if (t != (ssize_t) (sizeof(char)*tag.index()))
				SG_SERROR("Write error!\n")
		}

		temp[0] = '\n';
		t = write(f, temp, 1);
		if (t != 1)
			SG_SERROR("Write error!\n")
	}
}

void CVowpalWabbit::set_verbose(bool verbose)
{
	quiet=verbose==false;
}


float32_t CVowpalWabbit::compute_exact_norm(VwExample* &ex, float32_t& sum_abs_x)
{
	// We must traverse the features in _precisely_ the same order as during training.
	vw_size_t thread_mask = env->thread_mask;
	vw_size_t thread_num = 0;

	float32_t g = reg->loss->get_square_grad(ex->final_prediction, ex->ld->label) * ex->ld->weight;
	if (g == 0) return 0.;

	float32_t xGx = 0.;

	float32_t* weights = reg->weight_vectors[thread_num];
	for (vw_size_t* i = ex->indices.begin; i != ex->indices.end; i++)
	{
		for (VwFeature* f = ex->atomics[*i].begin; f != ex->atomics[*i].end; f++)
		{
			float32_t* w_vec = &weights[f->weight_index & thread_mask];
			float32_t t = f->x * CMath::invsqrt(w_vec[1] + g * f->x * f->x);
			xGx += t * f->x;
			sum_abs_x += fabsf(f->x);
		}
	}

	for (int32_t k = 0; k < env->pairs.get_num_elements(); k++)
	{
		char* i = env->pairs.get_element(k);

		v_array<VwFeature> temp = ex->atomics[(int32_t)(i[0])];
		temp.begin = ex->atomics[(int32_t)(i[0])].begin;
		temp.end = ex->atomics[(int32_t)(i[0])].end;
		for (; temp.begin != temp.end; temp.begin++)
			xGx += compute_exact_norm_quad(weights, *temp.begin, ex->atomics[(int32_t)(i[1])], thread_mask, g, sum_abs_x);
	}

	return xGx;
}

float32_t CVowpalWabbit::compute_exact_norm_quad(float32_t* weights, VwFeature& page_feature, v_array<VwFeature> &offer_features,
						 vw_size_t mask, float32_t g, float32_t& sum_abs_x)
{
	vw_size_t halfhash = quadratic_constant * page_feature.weight_index;
	float32_t xGx = 0.;
	float32_t update2 = g * page_feature.x * page_feature.x;
	for (VwFeature* elem = offer_features.begin; elem != offer_features.end; elem++)
	{
		float32_t* w_vec = &weights[(halfhash + elem->weight_index) & mask];
		float32_t t = elem->x * CMath::invsqrt(w_vec[1] + update2 * elem->x * elem->x);
		xGx += t * elem->x;
		sum_abs_x += fabsf(elem->x);
	}
	return xGx;
}
