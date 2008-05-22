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

#include "lib/config.h"

#ifndef HAVE_SWIG
#include <unistd.h>

#include "lib/common.h"

#include "guilib/GUIHMM.h"
#include "interface/SGInterface.h"

#include "features/StringFeatures.h"
#include "features/Labels.h"

#define TMP_DIR "/tmp/"


CGUIHMM::CGUIHMM(CSGInterface* ui_): CSGObject(), ui(ui_)
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

}

bool CGUIHMM::new_hmm(INT n, INT m)
{
	delete working;
	working=new CHMM(n, m, NULL, PSEUDO);
	M=m;
	return true;
}

bool CGUIHMM::baum_welch_train()
{
	if (!working)
		SG_ERROR("Create HMM first.\n");

	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	if (!trainfeatures)
		SG_ERROR("Assign train features first.\n");
	if (trainfeatures->get_feature_type()!=F_WORD ||
		trainfeatures->get_feature_class()!=C_STRING)
		SG_ERROR("Features must be STRING of type WORD.\n");

	CStringFeatures<WORD>* sf=(CStringFeatures<WORD>*) trainfeatures;
	SG_DEBUG("Stringfeatures have %ld orig_symbols %ld symbols %d order %ld max_symbols\n",  (LONG) sf->get_original_num_symbols(), (LONG) sf->get_num_symbols(), sf->get_order(), (LONG) sf->get_max_num_symbols());

	working->set_observations(sf);

	return working->baum_welch_viterbi_train(BW_NORMAL);
}


bool CGUIHMM::baum_welch_trans_train()
{
	if (!working)
		SG_ERROR("Create HMM first.\n");

	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	if (!trainfeatures)
		SG_ERROR("Assign train features first.\n");
	if (trainfeatures->get_feature_type()!=F_WORD ||
		trainfeatures->get_feature_class()!=C_STRING)
		SG_ERROR("Features must be STRING of type WORD.\n");

	working->set_observations((CStringFeatures<WORD>*) trainfeatures);

	return working->baum_welch_viterbi_train(BW_TRANS);
}


bool CGUIHMM::baum_welch_train_defined()
{
	if (!working)
		SG_ERROR("Create HMM first.\n");
	if (!working->get_observations())
		SG_ERROR("Assign observation first.\n");

	return working->baum_welch_viterbi_train(BW_DEFINED);
}

bool CGUIHMM::viterbi_train()
{
	if (!working)
		SG_ERROR("Create HMM first.\n");
	if (!working->get_observations())
		SG_ERROR("Assign observation first.\n");

	return working->baum_welch_viterbi_train(VIT_NORMAL);
}

bool CGUIHMM::viterbi_train_defined()
{
	if (!working)
		SG_ERROR("Create HMM first.\n");
	if (!working->get_observations())
		SG_ERROR("Assign observation first.\n");

	return working->baum_welch_viterbi_train(VIT_DEFINED);
}

bool CGUIHMM::linear_train(CHAR align)
{
	if (!working)
		SG_ERROR("Create HMM first.\n");

	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	if (!trainfeatures)
		SG_ERROR("Assign train features first.\n");
	if (trainfeatures->get_feature_type()!=F_WORD ||
		trainfeatures->get_feature_class()!=C_STRING)
		SG_ERROR("Features must be STRING of type WORD.\n");

	working->set_observations((CStringFeatures<WORD>*) ui->
		ui_features->get_train_features());

	bool right_align=false;
	if (align=='r')
	{
		SG_INFO("Using alignment to right.\n");
		right_align=true;
	}
	else
		SG_INFO("Using alignment to left.\n");
	working->linear_train(right_align);

	return true;
}

bool CGUIHMM::one_class_test(
	CHAR* filename_out, CHAR* filename_roc, bool is_linear)
{
	bool result=false;
	FILE* file_out=stdout;
	FILE* file_roc=NULL;

	if (filename_out)
	{
		file_out=fopen(filename_out, "w");

		if (!file_out)
			SG_ERROR("Could not open file %s.\n", filename_out);

		if (filename_roc)
		{
			file_roc=fopen(filename_roc, "w");

			if (!file_roc)
				SG_ERROR("Could not open %s.\n", filename_roc);
		}
	}

	if (!working)
		SG_ERROR("No HMM defined!\n");

	if (ui->ui_features->get_test_features())
		SG_ERROR("Assign posttest and negtest observations first!\n");

	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) ui->
		ui_features->get_test_features();
	working->set_observations(obs);
	CStringFeatures<WORD>* old_test=working->get_observations();
	CLabels* lab=ui->ui_labels->get_test_labels();
	INT total=obs->get_num_vectors();
	ASSERT(lab && total==lab->get_num_labels());
	DREAL* output=new DREAL[total];
	INT* label=new INT[total];

	for (INT dim=0; dim<total; dim++)
	{
		output[dim]= is_linear ? working->linear_model_probability(dim) : working->model_probability(dim);
		label[dim]= lab->get_int_label(dim);
	}

	ui->ui_math->evaluate_results(output, label, total, file_out, file_roc);
	working->set_observations(old_test);
	result=true;

	delete[] output;
	delete[] label;
	delete obs;

	if (file_roc)
		fclose(file_roc);
	if (file_out && file_out!=stdout)
		fclose(file_out);

	return result;
}

bool CGUIHMM::hmm_classify(CHAR* param)
{
	bool result=false;
	CHAR outputname[1024];
	CHAR rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	INT numargs=-1;
	INT poslinear=0;
	INT neglinear=0;

	param=CIO::skip_spaces(param);

	numargs=sscanf(param, "%s %d %d", outputname, &neglinear, &poslinear);

	if (numargs>=1)
	{
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
			SG_ERROR( "could not open %s\n",outputname);
			return false;
		}

		if (numargs>=2) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				SG_ERROR( "could not open %s\n",rocfname);
				return false;
			}
		}
	}

	if (pos && neg)
	{
		if (ui->ui_features->get_test_features())
		{
			CStringFeatures<WORD>* o= (CStringFeatures<WORD>*) ui->
				ui_features->get_test_features();
			CLabels* lab= ui->ui_labels->get_test_labels();

			//CStringFeatures<WORD>* old_pos=pos->get_observations();
			//CStringFeatures<WORD>* old_neg=neg->get_observations();

			pos->set_observations(o);
			neg->set_observations(o);

			INT total=o->get_num_vectors();

			DREAL* output = new DREAL[total];	
			INT* label= new INT[total];	

			SG_INFO( "classifying using neg %s hmm vs. pos %s hmm\n", neglinear ? "linear" : "", poslinear ? "linear" : "");

			for (INT dim=0; dim<total; dim++)
			{
				output[dim]= 
					(poslinear ? pos->linear_model_probability(dim) : pos->model_probability(dim)) -
					(neglinear ? neg->linear_model_probability(dim) : neg->model_probability(dim));
				label[dim]= lab->get_int_label(dim);
			}

			ui->ui_math->evaluate_results(output, label, total, outputfile);

			delete[] output;
			delete[] label;

			//pos->set_observations(old_pos);
			//neg->set_observations(old_neg);
			result=true;
		}
		else
			printf("load test features first!\n");
	}
	else
		SG_ERROR( "assign positive and negative models first!\n");

	if ((outputfile) && (outputfile!=stdout))
		fclose(outputfile);

	return result;
}

bool CGUIHMM::hmm_test(
	CHAR* filename_out, CHAR* filename_roc,
	bool is_pos_linear, bool is_neg_linear)
{
	bool result=false;
	FILE* file_output=stdout;
	FILE* file_roc=NULL;

	if (filename_out)
	{
		file_output=fopen(filename_out, "w");

		if (!file_output)
			SG_ERROR("Could not open file %s.\n", filename_out);

		if (filename_roc)
		{
			file_roc=fopen(filename_roc, "w");

			if (!file_roc)
				SG_ERROR("Could not open file %s.\n", filename_roc);
		}
	}

	if (!(pos && neg))
		SG_ERROR("Assign positive and negative models first!\n");

	if (!ui->ui_features->get_test_features())
		SG_ERROR("Assign test features first!\n");

	CStringFeatures<WORD>* o=(CStringFeatures<WORD>*) ui->
		ui_features->get_test_features();
	ASSERT(o);
	CLabels* lab=ui->ui_labels->get_test_labels();
	CStringFeatures<WORD>* old_pos=pos->get_observations();
	CStringFeatures<WORD>* old_neg=neg->get_observations();
	pos->set_observations(o);
	neg->set_observations(o);
	INT total=o->get_num_vectors();
	ASSERT(lab && total==lab->get_num_labels());
	DREAL* output=new DREAL[total];
	INT* label=new INT[total];

	SG_INFO("Testing using neg %s hmm vs. pos %s hmm\n", is_neg_linear ? "linear" : "", is_pos_linear ? "linear" : "");

	for (INT dim=0; dim<total; dim++)
	{
		output[dim]=
			(is_pos_linear ? pos->linear_model_probability(dim) : pos->model_probability(dim)) -
			(is_neg_linear ? neg->linear_model_probability(dim) : neg->model_probability(dim));
		label[dim]= lab->get_int_label(dim);
		//fprintf(file_output, "%+d: %f - %f = %f\n", label[dim], pos->model_probability(dim), neg->model_probability(dim), output[dim]);
	}

	ui->ui_math->evaluate_results(output, label, total, file_output, file_roc);

	delete[] output;
	delete[] label;

	pos->set_observations(old_pos);
	neg->set_observations(old_neg);

	result=true;

	if (file_roc)
		fclose(file_roc);
	if (file_output && file_output!=stdout)
		fclose(file_output);

	return result;
}

CLabels* CGUIHMM::classify(CLabels* result)
{
	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) ui->
		ui_features->get_test_features();
	ASSERT(obs);
	INT num_vec=obs->get_num_vectors();

	//CStringFeatures<WORD>* old_pos=pos->get_observations();
	//CStringFeatures<WORD>* old_neg=neg->get_observations();

	pos->set_observations(obs);
	neg->set_observations(obs);

	if (!result)
		result=new CLabels(num_vec);

	for (INT i=0; i<num_vec; i++)
		result->set_label(i, pos->model_probability(i) - neg->model_probability(i));

	//pos->set_observations(old_pos);
	//neg->set_observations(old_neg);
	return result;
}

DREAL CGUIHMM::classify_example(INT idx)
{
	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) ui->
		ui_features->get_test_features();
	ASSERT(obs);

	//CStringFeatures<WORD>* old_pos=pos->get_observations();
	//CStringFeatures<WORD>* old_neg=neg->get_observations();

	pos->set_observations(obs);
	neg->set_observations(obs);

	DREAL result=pos->model_probability(idx) - neg->model_probability(idx);
	//pos->set_observations(old_pos);
	//neg->set_observations(old_neg);
	return result;
}

CLabels* CGUIHMM::one_class_classify(CLabels* result)
{
	ASSERT(working);

	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) ui->
		ui_features->get_test_features();
	ASSERT(obs);
	INT num_vec=obs->get_num_vectors();

	//CStringFeatures<WORD>* old_pos=working->get_observations();
	working->set_observations(obs);

	if (!result)
		result=new CLabels(num_vec);

	for (INT i=0; i<num_vec; i++)
		result->set_label(i, working->model_probability(i));

	//working->set_observations(old_pos);
	return result;
}

CLabels* CGUIHMM::linear_one_class_classify(CLabels* result)
{
	ASSERT(working);

	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) ui->
		ui_features->get_test_features();
	ASSERT(obs);
	INT num_vec=obs->get_num_vectors();

	//CStringFeatures<WORD>* old_pos=working->get_observations();
	working->set_observations(obs);

	if (!result)
		result=new CLabels(num_vec);

	for (INT i=0; i<num_vec; i++)
		result->set_label(i, working->linear_model_probability(i));

	//working->set_observations(old_pos);
	return result;
}


DREAL CGUIHMM::one_class_classify_example(INT idx)
{
	ASSERT(working);

	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) ui->
		ui_features->get_test_features();
	ASSERT(obs);

	//CStringFeatures<WORD>* old_pos=pos->get_observations();

	pos->set_observations(obs);
	neg->set_observations(obs);

	DREAL result=working->model_probability(idx);
	//working->set_observations(old_pos);
	return result;
}

bool CGUIHMM::append_model(CHAR* param)
{
	if (working)
	{
		CHAR fname[1024]; 
		INT base1=0;
		INT base2=2;
		param=CIO::skip_spaces(param);

		INT num_param=sscanf(param, "%s %i %i", fname, &base1, &base2);

		if (num_param==3 || num_param==1)
		{
			FILE* model_file=fopen(fname, "r");

			if (model_file)
			{

				CHMM* h=new CHMM(model_file,PSEUDO);
				if (h && h->get_status())
				{
					printf("file successfully read\n");
					fclose(model_file);

					DREAL* cur_o=new DREAL[h->get_M()];
					DREAL* app_o=new DREAL[h->get_M()];
					ASSERT(cur_o && app_o);

					SG_DEBUG( "h %d , M: %d\n", h, h->get_M());

					for (INT i=0; i<h->get_M(); i++)
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

					if (num_param==3)
						working->append_model(h, cur_o, app_o);
					else
						working->append_model(h);

					delete[] cur_o;
					delete[] app_o;
					SG_INFO( "new model has %i states\n", working->get_N());
					delete h;
				}
				else
					SG_ERROR( "reading file %s failed\n", fname);
			}
			else
				SG_ERROR( "opening file %s failed\n", fname);
		}
		else
			SG_ERROR( "see help for parameters\n", fname);
	}
	else
		SG_ERROR( "create model first\n");


	return false;
}

bool CGUIHMM::add_states(INT num_states, DREAL value)
{
	if (!working)
		SG_ERROR("Create HMM first.\n");

	working->add_states(num_states, value);
	SG_INFO("New model has %i states, value %f.\n", working->get_N(), value);
	return true;
}

bool CGUIHMM::set_pseudo(DREAL pseudo)
{
	PSEUDO=pseudo;
	SG_INFO("Current setting: pseudo=%e.\n", PSEUDO);
	return true;
}

bool CGUIHMM::convergence_criteria(INT num_iterations, DREAL epsilon)
{
	if (!working)
		SG_ERROR("Create HMM first.\n");

	working->set_iterations(num_iterations);
	working->set_epsilon(epsilon);

	SG_INFO("Current HMM convergence criteria: iterations=%i, epsilon=%e\n", working->get_iterations(), working->get_epsilon());
	return true;
}

bool CGUIHMM::set_hmm_as(CHAR* target)
{
	if (!working)
		SG_ERROR("Create HMM first!\n");

	if (strncmp(target, "POS", 3)==0)
	{
		delete pos;
		pos=working;
		working=NULL;
	}
	else if (strncmp(target, "NEG", 3)==0)
	{
		delete neg;
		neg=working;
		working=NULL;
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		delete test;
		test=working;
		working=NULL;
	}
	else
		SG_ERROR("Target POS|NEG|TEST is missing.\n");

	return true;
}

bool CGUIHMM::load(CHAR* filename)
{
	bool result=false;

	FILE* model_file=fopen(filename, "r");
	if (!model_file)
		SG_ERROR("Opening file %s failed.\n", filename);

	delete working;
	working=new CHMM(model_file, PSEUDO);
	fclose(model_file);

	if (working && working->get_status())
	{
		SG_INFO("Loaded HMM successfully from file %s.\n", filename);
		result=true;
	}

	M=working->get_M();

	return result;
}

bool CGUIHMM::save(CHAR* filename, bool is_binary)
{
	bool result=false;

	if (!working)
		SG_ERROR("Create HMM first.\n");

	FILE* file=fopen(filename, "w");
	if (file)
	{
		if (is_binary)
			result=working->save_model_bin(file);
		else
			result=working->save_model(file);
	}

	if (!file || !result)
		SG_ERROR("Writing to file %s failed!\n", filename);
	else
		SG_INFO("Successfully written model into %s!\n", filename);

	if (file)
		fclose(file);

	return result;
}

bool CGUIHMM::load_definitions(CHAR* filename, bool do_init)
{
	if (!working)
		SG_ERROR("Create HMM first.\n");

	bool result=false;
	FILE* def_file=fopen(filename, "r");
	if (!def_file)
		SG_ERROR("Opening file %s failed\n", filename);
	
	if (working->load_definitions(def_file, true, do_init))
	{
		SG_INFO("Definitions successfully read from %s.\n", filename);
		result=true;
	}
	else
		SG_ERROR("Couldn't load definitions form file %s.\n", filename);

	fclose(def_file);
	return result;
}

bool CGUIHMM::save_likelihood(CHAR* filename, bool is_binary)
{
	bool result=false;

	if (!working)
		SG_ERROR("Create HMM first\n");

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
		SG_ERROR("Writing to file %s failed!\n", filename);
	else
		SG_INFO("Successfully written likelihoods into %s!\n", filename);

	if (file)
		fclose(file);

	return result;
}

bool CGUIHMM::save_path(CHAR* filename, bool is_binary)
{
	bool result=false;
	if (!working)
		SG_ERROR("Create HMM first.\n");

	FILE* file=fopen(filename, "w");
	if (file)
	{
		/// ..future
		//if (binary)
		//_train()/	result=working->save_model_bin(file);
		//else
		CStringFeatures<WORD>* obs=(CStringFeatures<WORD>*) ui->
			ui_features->get_test_features();
		ASSERT(obs);
		working->set_observations(obs);

		result=working->save_path(file);
	}

	if (!file || !result)
		SG_ERROR("Writing to file %s failed!\n", filename);
	else
		SG_INFO("Successfully written path into %s!\n", filename);

	if (file)
		fclose(file);

	return result;
}

bool CGUIHMM::chop(DREAL value)
{
	if (!working)
		SG_ERROR("Create HMM first.\n");

	working->chop(value);
	return true;
}

bool CGUIHMM::likelihood()
{
	if (!working)
		SG_ERROR("Create HMM first!\n");

	working->output_model(false);
	return true;
}

bool CGUIHMM::output_hmm()
{
	if (!working)
		SG_ERROR("Create HMM first!\n");

	working->output_model(true);
	return true;
}

bool CGUIHMM::output_hmm_defined()
{
	if (!working)
		SG_ERROR("Create HMM first!\n");

	working->output_model_defined(true);
	return true;
}

bool CGUIHMM::best_path(INT from, INT to)
{
	// FIXME: from unused???

	if (!working)
		SG_ERROR("Create HMM first.\n");

	//get path
	working->best_path(0);

	for (INT t=0; t<working->get_observations()->get_vector_length(0)-1 && t<to; t++)
		SG_PRINT("%d ", working->get_best_path_state(0, t));
	SG_PRINT("\n");

	//for (t=0; t<p_observations->get_vector_length(0)-1 && t<to; t++)
	//	SG_PRINT( "%d ", PATH(0)[t]);
	//
	return true;
}

bool CGUIHMM::normalize(bool keep_dead_states)
{
	if (!working)
		SG_ERROR("Create HMM first.\n");

	working->normalize(keep_dead_states);
	return true;
}

bool CGUIHMM::relative_entropy(DREAL*& values, INT& len)
{
	if (!pos || !neg)
		SG_ERROR("Set pos and neg HMM first!\n");

	INT pos_N=pos->get_N();
	INT neg_N=neg->get_N();
	INT pos_M=pos->get_M();
	INT neg_M=neg->get_M();
	if (pos_M!=neg_M || pos_N!=neg_N)
		SG_ERROR("Pos and neg HMM's differ in number of emissions or states.\n");

	DREAL* p=new DREAL[pos_M];
	DREAL* q=new DREAL[neg_M];

	delete[] values;
	values=new DREAL[pos_N];

	for (INT i=0; i<pos_N; i++)
	{
		for (INT j=0; j<pos_M; j++)
		{
			p[j]=pos->get_b(i, j);
			q[j]=neg->get_b(i, j);
		}

		values[i]=CMath::relative_entropy(p, q, pos_M);
	}
	delete[] p;
	delete[] q;

	len=pos_N;
	return true;
}

bool CGUIHMM::entropy(DREAL*& values, INT& len)
{
	if (!working)
		SG_ERROR("Create HMM first!\n");

	INT n=working->get_N();
	INT m=working->get_M();
	DREAL* p=new DREAL[m];

	delete[] values;
	values=new DREAL[n];

	for (INT i=0; i<n; i++)
	{
		for (INT j=0; j<m; j++)
			p[j]=working->get_b(i, j);

		values[i]=CMath::entropy(p, m);
	}
	delete[] p;

	len=m;
	return true;
}

bool CGUIHMM::permutation_entropy(INT width, INT seq_num)
{
	if (!working)
		SG_ERROR("Create hmm first.\n");

	if (!working->get_observations())
		SG_ERROR("Set observations first.\n");

	return working->permutation_entropy(width, seq_num);
}
#endif
