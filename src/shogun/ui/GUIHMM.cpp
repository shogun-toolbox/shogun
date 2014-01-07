/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <ui/GUIHMM.h>
#include <ui/SGInterface.h>

#include <lib/config.h>
#include <lib/common.h>
#include <features/StringFeatures.h>
#include <labels/Labels.h>
#include <labels/RegressionLabels.h>
#include <mathematics/Statistics.h>

#include <unistd.h>

using namespace shogun;

CGUIHMM::CGUIHMM(CSGInterface* ui_)
: CSGObject(), ui(ui_)
{
	working=NULL;

	pos=NULL;
	neg=NULL;
	test=NULL;

	PSEUDO=1e-10;
	M=4;
}

CGUIHMM::~CGUIHMM()
{
	SG_UNREF(working);
}

bool CGUIHMM::new_hmm(int32_t n, int32_t m)
{
	SG_UNREF(working);
	working=new CHMM(n, m, NULL, PSEUDO);
	M=m;
	return true;
}

bool CGUIHMM::baum_welch_train()
{
	if (!working)
		SG_ERROR("Create HMM first.\n")

	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	if (!trainfeatures)
		SG_ERROR("Assign train features first.\n")
	if (trainfeatures->get_feature_type()!=F_WORD ||
		trainfeatures->get_feature_class()!=C_STRING)
		SG_ERROR("Features must be STRING of type WORD.\n")

	CStringFeatures<uint16_t>* sf=(CStringFeatures<uint16_t>*) trainfeatures;
	SG_DEBUG("Stringfeatures have %ld orig_symbols %ld symbols %d order %ld max_symbols\n",  (int64_t) sf->get_original_num_symbols(), (int64_t) sf->get_num_symbols(), sf->get_order(), (int64_t) sf->get_max_num_symbols())

	working->set_observations(sf);

	return working->baum_welch_viterbi_train(BW_NORMAL);
}


bool CGUIHMM::baum_welch_trans_train()
{
	if (!working)
		SG_ERROR("Create HMM first.\n")

	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	if (!trainfeatures)
		SG_ERROR("Assign train features first.\n")
	if (trainfeatures->get_feature_type()!=F_WORD ||
		trainfeatures->get_feature_class()!=C_STRING)
		SG_ERROR("Features must be STRING of type WORD.\n")

	working->set_observations((CStringFeatures<uint16_t>*) trainfeatures);

	return working->baum_welch_viterbi_train(BW_TRANS);
}


bool CGUIHMM::baum_welch_train_defined()
{
	if (!working)
		SG_ERROR("Create HMM first.\n")
	if (!working->get_observations())
		SG_ERROR("Assign observation first.\n")

	return working->baum_welch_viterbi_train(BW_DEFINED);
}

bool CGUIHMM::viterbi_train()
{
	if (!working)
		SG_ERROR("Create HMM first.\n")
	if (!working->get_observations())
		SG_ERROR("Assign observation first.\n")

	return working->baum_welch_viterbi_train(VIT_NORMAL);
}

bool CGUIHMM::viterbi_train_defined()
{
	if (!working)
		SG_ERROR("Create HMM first.\n")
	if (!working->get_observations())
		SG_ERROR("Assign observation first.\n")

	return working->baum_welch_viterbi_train(VIT_DEFINED);
}

bool CGUIHMM::linear_train(char align)
{
	if (!working)
		SG_ERROR("Create HMM first.\n")

	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	if (!trainfeatures)
		SG_ERROR("Assign train features first.\n")
	if (trainfeatures->get_feature_type()!=F_WORD ||
		trainfeatures->get_feature_class()!=C_STRING)
		SG_ERROR("Features must be STRING of type WORD.\n")

	working->set_observations((CStringFeatures<uint16_t>*) ui->
		ui_features->get_train_features());

	bool right_align=false;
	if (align=='r')
	{
		SG_INFO("Using alignment to right.\n")
		right_align=true;
	}
	else
		SG_INFO("Using alignment to left.\n")
	working->linear_train(right_align);

	return true;
}

CRegressionLabels* CGUIHMM::classify(CRegressionLabels* result)
{
	CStringFeatures<uint16_t>* obs= (CStringFeatures<uint16_t>*) ui->
		ui_features->get_test_features();
	ASSERT(obs)
	int32_t num_vec=obs->get_num_vectors();

	//CStringFeatures<uint16_t>* old_pos=pos->get_observations();
	//CStringFeatures<uint16_t>* old_neg=neg->get_observations();

	pos->set_observations(obs);
	neg->set_observations(obs);

	if (!result)
		result=new CRegressionLabels(num_vec);

	for (int32_t i=0; i<num_vec; i++)
		result->set_label(i, pos->model_probability(i) - neg->model_probability(i));

	//pos->set_observations(old_pos);
	//neg->set_observations(old_neg);
	return result;
}

float64_t CGUIHMM::classify_example(int32_t idx)
{
	CStringFeatures<uint16_t>* obs= (CStringFeatures<uint16_t>*) ui->
		ui_features->get_test_features();
	ASSERT(obs)

	//CStringFeatures<uint16_t>* old_pos=pos->get_observations();
	//CStringFeatures<uint16_t>* old_neg=neg->get_observations();

	pos->set_observations(obs);
	neg->set_observations(obs);

	float64_t result=pos->model_probability(idx) - neg->model_probability(idx);
	//pos->set_observations(old_pos);
	//neg->set_observations(old_neg);
	return result;
}

CRegressionLabels* CGUIHMM::one_class_classify(CRegressionLabels* result)
{
	ASSERT(working)

	CStringFeatures<uint16_t>* obs= (CStringFeatures<uint16_t>*) ui->
		ui_features->get_test_features();
	ASSERT(obs)
	int32_t num_vec=obs->get_num_vectors();

	//CStringFeatures<uint16_t>* old_pos=working->get_observations();
	working->set_observations(obs);

	if (!result)
		result=new CRegressionLabels(num_vec);

	for (int32_t i=0; i<num_vec; i++)
		result->set_label(i, working->model_probability(i));

	//working->set_observations(old_pos);
	return result;
}

CRegressionLabels* CGUIHMM::linear_one_class_classify(CRegressionLabels* result)
{
	ASSERT(working)

	CStringFeatures<uint16_t>* obs= (CStringFeatures<uint16_t>*) ui->
		ui_features->get_test_features();
	ASSERT(obs)
	int32_t num_vec=obs->get_num_vectors();

	//CStringFeatures<uint16_t>* old_pos=working->get_observations();
	working->set_observations(obs);

	if (!result)
		result=new CRegressionLabels(num_vec);

	for (int32_t i=0; i<num_vec; i++)
		result->set_label(i, working->linear_model_probability(i));

	//working->set_observations(old_pos);
	return result;
}


float64_t CGUIHMM::one_class_classify_example(int32_t idx)
{
	ASSERT(working)

	CStringFeatures<uint16_t>* obs= (CStringFeatures<uint16_t>*) ui->
		ui_features->get_test_features();
	ASSERT(obs)

	//CStringFeatures<uint16_t>* old_pos=pos->get_observations();

	pos->set_observations(obs);
	neg->set_observations(obs);

	float64_t result=working->model_probability(idx);
	//working->set_observations(old_pos);
	return result;
}

bool CGUIHMM::append_model(char* filename, int32_t base1, int32_t base2)
{
	if (!working)
		SG_ERROR("Create HMM first.\n")
	if (!filename)
		SG_ERROR("Invalid filename.\n")

	FILE* model_file=fopen(filename, "r");
	if (!model_file)
		SG_ERROR("Opening file %s failed.\n", filename)

	CHMM* h=new CHMM(model_file,PSEUDO);
	if (!h || !h->get_status())
	{
		SG_UNREF(h);
		fclose(model_file);
		SG_ERROR("Reading file %s failed.\n", filename)
	}

	fclose(model_file);
	SG_INFO("File %s successfully read.\n", filename)

	SG_DEBUG("h %d , M: %d\n", h, h->get_M())
	if (base1!=-1 && base2!=-1)
	{
		float64_t* cur_o=SG_MALLOC(float64_t, h->get_M());
		float64_t* app_o=SG_MALLOC(float64_t, h->get_M());

		for (int32_t i=0; i<h->get_M(); i++)
		{
			if (i==base1)
				cur_o[i]=0;
			else
				cur_o[i]=-1000;

			if (i==base2)
				app_o[i]=0;
			else
				app_o[i]=-1000;
		}

		working->append_model(h, cur_o, app_o);

		SG_FREE(cur_o);
		SG_FREE(app_o);
	}
	else
		working->append_model(h);

	SG_UNREF(h);
	SG_INFO("New model has %i states.\n", working->get_N())
	return true;
}

bool CGUIHMM::add_states(int32_t num_states, float64_t value)
{
	if (!working)
		SG_ERROR("Create HMM first.\n")

	working->add_states(num_states, value);
	SG_INFO("New model has %i states, value %f.\n", working->get_N(), value)
	return true;
}

bool CGUIHMM::set_pseudo(float64_t pseudo)
{
	PSEUDO=pseudo;
	SG_INFO("Current setting: pseudo=%e.\n", PSEUDO)
	return true;
}

bool CGUIHMM::convergence_criteria(int32_t num_iterations, float64_t epsilon)
{
	if (!working)
		SG_ERROR("Create HMM first.\n")

	working->set_iterations(num_iterations);
	working->set_epsilon(epsilon);

	SG_INFO("Current HMM convergence criteria: iterations=%i, epsilon=%e\n", working->get_iterations(), working->get_epsilon())
	return true;
}

bool CGUIHMM::set_hmm_as(char* target)
{
	if (!working)
		SG_ERROR("Create HMM first!\n")

	if (strncmp(target, "POS", 3)==0)
	{
		SG_UNREF(pos);
		pos=working;
		working=NULL;
	}
	else if (strncmp(target, "NEG", 3)==0)
	{
		SG_UNREF(neg);
		neg=working;
		working=NULL;
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		SG_UNREF(test);
		test=working;
		working=NULL;
	}
	else
		SG_ERROR("Target POS|NEG|TEST is missing.\n")

	return true;
}

bool CGUIHMM::load(char* filename)
{
	bool result=false;

	FILE* model_file=fopen(filename, "r");
	if (!model_file)
		SG_ERROR("Opening file %s failed.\n", filename)

	SG_UNREF(working);
	working=new CHMM(model_file, PSEUDO);
	fclose(model_file);

	if (working && working->get_status())
	{
		SG_INFO("Loaded HMM successfully from file %s.\n", filename)
		result=true;
	}

	M=working->get_M();

	return result;
}

bool CGUIHMM::save(char* filename, bool is_binary)
{
	bool result=false;

	if (!working)
		SG_ERROR("Create HMM first.\n")

	FILE* file=fopen(filename, "w");
	if (file)
	{
		if (is_binary)
			result=working->save_model_bin(file);
		else
			result=working->save_model(file);
	}

	if (!file || !result)
		SG_ERROR("Writing to file %s failed!\n", filename)
	else
		SG_INFO("Successfully written model into %s!\n", filename)

	if (file)
		fclose(file);

	return result;
}

bool CGUIHMM::load_definitions(char* filename, bool do_init)
{
	if (!working)
		SG_ERROR("Create HMM first.\n")

	bool result=false;
	FILE* def_file=fopen(filename, "r");
	if (!def_file)
		SG_ERROR("Opening file %s failed\n", filename)

	if (working->load_definitions(def_file, true, do_init))
	{
		SG_INFO("Definitions successfully read from %s.\n", filename)
		result=true;
	}
	else
		SG_ERROR("Couldn't load definitions form file %s.\n", filename)

	fclose(def_file);
	return result;
}

bool CGUIHMM::save_likelihood(char* filename, bool is_binary)
{
	bool result=false;

	if (!working)
		SG_ERROR("Create HMM first\n")

	FILE* file=fopen(filename, "w");
	if (file)
	{
		/// ..future
		//if (binary)
		//	result=working->save_model_bin(file);
		//else

		result=working->save_likelihood(file);
	}

	if (!file || !result)
		SG_ERROR("Writing to file %s failed!\n", filename)
	else
		SG_INFO("Successfully written likelihoods into %s!\n", filename)

	if (file)
		fclose(file);

	return result;
}

bool CGUIHMM::save_path(char* filename, bool is_binary)
{
	bool result=false;
	if (!working)
		SG_ERROR("Create HMM first.\n")

	FILE* file=fopen(filename, "w");
	if (file)
	{
		/// ..future
		//if (binary)
		//_train()/	result=working->save_model_bin(file);
		//else
		CStringFeatures<uint16_t>* obs=(CStringFeatures<uint16_t>*) ui->
			ui_features->get_test_features();
		ASSERT(obs)
		working->set_observations(obs);

		result=working->save_path(file);
	}

	if (!file || !result)
		SG_ERROR("Writing to file %s failed!\n", filename)
	else
		SG_INFO("Successfully written path into %s!\n", filename)

	if (file)
		fclose(file);

	return result;
}

bool CGUIHMM::chop(float64_t value)
{
	if (!working)
		SG_ERROR("Create HMM first.\n")

	working->chop(value);
	return true;
}

bool CGUIHMM::likelihood()
{
	if (!working)
		SG_ERROR("Create HMM first!\n")

	working->output_model(false);
	return true;
}

bool CGUIHMM::output_hmm()
{
	if (!working)
		SG_ERROR("Create HMM first!\n")

	working->output_model(true);
	return true;
}

bool CGUIHMM::output_hmm_defined()
{
	if (!working)
		SG_ERROR("Create HMM first!\n")

	working->output_model_defined(true);
	return true;
}

bool CGUIHMM::best_path(int32_t from, int32_t to)
{
	// FIXME: from unused???

	if (!working)
		SG_ERROR("Create HMM first.\n")

	//get path
	working->best_path(0);

	for (int32_t t=0; t<working->get_observations()->get_vector_length(0)-1 && t<to; t++)
		SG_PRINT("%d ", working->get_best_path_state(0, t))
	SG_PRINT("\n")

	//for (t=0; t<p_observations->get_vector_length(0)-1 && t<to; t++)
	//	SG_PRINT("%d ", PATH(0)[t])
	//
	return true;
}

bool CGUIHMM::normalize(bool keep_dead_states)
{
	if (!working)
		SG_ERROR("Create HMM first.\n")

	working->normalize(keep_dead_states);
	return true;
}

bool CGUIHMM::relative_entropy(float64_t*& values, int32_t& len)
{
	if (!pos || !neg)
		SG_ERROR("Set pos and neg HMM first!\n")

	int32_t pos_N=pos->get_N();
	int32_t neg_N=neg->get_N();
	int32_t pos_M=pos->get_M();
	int32_t neg_M=neg->get_M();
	if (pos_M!=neg_M || pos_N!=neg_N)
		SG_ERROR("Pos and neg HMM's differ in number of emissions or states.\n")

	float64_t* p=SG_MALLOC(float64_t, pos_M);
	float64_t* q=SG_MALLOC(float64_t, neg_M);

	SG_FREE(values);
	values=SG_MALLOC(float64_t, pos_N);

	for (int32_t i=0; i<pos_N; i++)
	{
		for (int32_t j=0; j<pos_M; j++)
		{
			p[j]=pos->get_b(i, j);
			q[j]=neg->get_b(i, j);
		}

		values[i]=CStatistics::relative_entropy(p, q, pos_M);
	}
	SG_FREE(p);
	SG_FREE(q);

	len=pos_N;
	return true;
}

bool CGUIHMM::entropy(float64_t*& values, int32_t& len)
{
	if (!working)
		SG_ERROR("Create HMM first!\n")

	int32_t n=working->get_N();
	int32_t m=working->get_M();
	float64_t* p=SG_MALLOC(float64_t, m);

	SG_FREE(values);
	values=SG_MALLOC(float64_t, n);

	for (int32_t i=0; i<n; i++)
	{
		for (int32_t j=0; j<m; j++)
			p[j]=working->get_b(i, j);

		values[i]=CStatistics::entropy(p, m);
	}
	SG_FREE(p);

	len=m;
	return true;
}

bool CGUIHMM::permutation_entropy(int32_t width, int32_t seq_num)
{
	if (!working)
		SG_ERROR("Create hmm first.\n")

	if (!working->get_observations())
		SG_ERROR("Set observations first.\n")

	return working->permutation_entropy(width, seq_num);
}
