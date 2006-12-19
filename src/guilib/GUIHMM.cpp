/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef HAVE_SWIG
#include <unistd.h>
#include "lib/common.h"
#include "guilib/GUIHMM.h"
#include "gui/GUI.h"
#include "features/StringFeatures.h"
#include "features/Labels.h"

CGUIHMM::CGUIHMM(CGUI * gui_): gui(gui_)
{
#ifdef PARALLEL
	number_of_hmm_tables = sysconf( _SC_NPROCESSORS_ONLN );
#else
	number_of_hmm_tables=1 ;
#endif
	working=NULL;

	pos=NULL;
	neg=NULL;
	test=NULL;

	ITERATIONS=150;
	EPSILON=1e-4;
	PSEUDO=1e-10;
	M=4;
	conv_it=5;
}

CGUIHMM::~CGUIHMM()
{

}

bool CGUIHMM::set_num_hmm_tables(CHAR* param)
{
	param=CIO::skip_spaces(param);

	INT tmp;
	if (sscanf(param, "%d", &tmp) == 1)
	{
		if (tmp>0)
		{
			number_of_hmm_tables=tmp ;
			CIO::message(M_INFO, "using %i separate tables\n",number_of_hmm_tables) ;
			return true ;
		} ;
	} ;

	return false;
}

bool CGUIHMM::new_hmm(CHAR* param)
{
	param=CIO::skip_spaces(param);

	INT n,m;
	if (sscanf(param, "%d %d", &n, &m) == 2)
	{
		if (working)
			delete working;

		working=new CHMM(n,m,NULL,PSEUDO, number_of_hmm_tables);
		M=m;
		return true;
	}
	else
		CIO::message(M_ERROR, "see help for parameters\n");

	return false;
}

bool CGUIHMM::baum_welch_train(CHAR* param)
{
	CHAR templname[]=TMP_DIR "bw_model_XXXXXX" ;
#if defined SUNOS || defined CYGWIN
#define mkstemp(name) mktemp(name);
#endif
	if ((gui->guifeatures.get_train_features()->get_feature_type()
	     !=F_WORD) ||
	   (gui->guifeatures.get_train_features()->get_feature_class()
	    !=C_STRING))
	  {
	    CIO::message(M_ERROR, "Features must be STRING of type WORD\n") ;
	    return false ;
	  } ;
	CStringFeatures<WORD>* sf = ((CStringFeatures<WORD>*) (gui->guifeatures.get_train_features()));
	CIO::message(M_DEBUG, "Stringfeatures have %d orig_symbols %d symbols %d order %d max_symbols\n",  sf->get_original_num_symbols(), sf->get_num_symbols(), sf->get_order(), sf->get_max_num_symbols());

	mkstemp(templname);
	CHAR templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
	iteration_count=ITERATIONS ;

	if (working) 
	{
		working->set_observations(sf);
		CHMM* working_estimate=new CHMM(working,number_of_hmm_tables);

		double prob_train=CMath::ALMOST_NEG_INFTY, prob = -CMath::INFTY ;

		while (!converge(prob,prob_train))
		{
			switch_model(&working, &working_estimate);
			prob=prob_train ;
			working->estimate_model_baum_welch(working_estimate);
			prob_train=working_estimate->model_probability();
			if (prob_max<prob_train)
			{
				prob_max=prob_train ;
#ifdef TMP_SAVE
				FILE* file=fopen(templname_best, "w");
				CIO::message(M_INFO, "\nsaving best model with filename %s ... ", templname_best) ;
				working->save_model(file) ;
				fclose(file) ;
				CIO::message(M_INFO, "done.") ;
#endif
			} 
			else
			{
#ifdef TMP_SAVE
				FILE* file=fopen(templname, "w");
				CIO::message(M_INFO, "\nsaving model with filename %s ... ", templname) ;
				working->save_model(file) ;
				fclose(file) ;
				CIO::message(M_INFO, "done.") ;
#endif
			}
		}
		delete working_estimate;
		working_estimate=NULL;
	}
	else
		CIO::message(M_ERROR, "create hmm first\n");

	return false;
}


bool CGUIHMM::baum_welch_trans_train(CHAR* param)
{
  if ((gui->guifeatures.get_train_features()->get_feature_type()
       !=F_WORD) ||
      (gui->guifeatures.get_train_features()->get_feature_class()
       !=C_STRING))
    {
      CIO::message(M_ERROR, "Features must be STRING of type WORD\n") ;
      return false ;
    } ;
  
  double prob_max=-CMath::INFTY ;
  iteration_count=ITERATIONS ;

  if (working) 
    {
      if (gui->guifeatures.get_train_features())
	{
	  working->set_observations((CStringFeatures<WORD>*) gui->guifeatures.get_train_features());
	  CHMM* working_estimate=new CHMM(working,number_of_hmm_tables);
	  
	  double prob_train=CMath::ALMOST_NEG_INFTY, prob = -CMath::INFTY ;
	  
	  while (!converge(prob,prob_train))
	    {
	      switch_model(&working, &working_estimate);
	      prob=prob_train ;
	      working->estimate_model_baum_welch_trans(working_estimate);
	      prob_train=working_estimate->model_probability();
	      if (prob_max<prob_train)
		{
		  prob_max=prob_train ;
		} ;
	    }
	  delete working_estimate;
	  working_estimate=NULL;
	}
      else
	CIO::message(M_ERROR, "load train features first\n");
    }
  else
    CIO::message(M_ERROR, "create model first\n");
  
  return false;
}


bool CGUIHMM::baum_welch_train_defined(CHAR* param)
{
	CHAR templname[]=TMP_DIR "bwdef_model_XXXXXX" ;
#ifdef SUNOS
#define mkstemp(name) mktemp(name);
#endif
	mkstemp(templname);
	CHAR templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
	iteration_count=ITERATIONS ;

	if (working) 
	{
		if (working->get_observations())
		{
			CHMM* working_estimate=new CHMM(working,number_of_hmm_tables);

			double prob_train=CMath::ALMOST_NEG_INFTY, prob = -CMath::INFTY ;

			while (!converge(prob,prob_train))
			{
				switch_model(&working, &working_estimate);
				prob=prob_train ;
				working->estimate_model_baum_welch_defined(working_estimate);
				prob_train=working_estimate->model_probability();
				if (prob_max<prob_train)
				{
					prob_max=prob_train ;
#ifdef TMP_SAVE
					FILE* file=fopen(templname_best, "w");
					CIO::message(M_INFO, "\nsaving best model with filename %s ... ", templname_best) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message(M_INFO, "done.") ;
#endif
				} 
				else
				{
#ifdef TMP_SAVE
					FILE* file=fopen(templname, "w");
					CIO::message(M_INFO, "\nsaving model with filename %s ... ", templname) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message(M_INFO, "done.") ;
#endif
				}
			}
			delete working_estimate;
			working_estimate=NULL;
		}
		else
			CIO::message(M_ERROR, "assign observation first\n");
	}
	else
		CIO::message(M_ERROR, "create model first\n");

	return false;
}

bool CGUIHMM::viterbi_train(CHAR* param)
{
	CHAR* templname= TMP_DIR "vit_model_XXXXXX" ;
#ifdef SUNOS
#define mkstemp(name) mktemp(name);
#endif
	mkstemp(templname);
	CHAR templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
	iteration_count=ITERATIONS ;

	if (working) 
	{
		if (working->get_observations())
		{
			CHMM* working_estimate=new CHMM(working,number_of_hmm_tables);

			double prob_train=CMath::ALMOST_NEG_INFTY, prob = -CMath::INFTY ;

			while (!converge(prob,prob_train))
			{
				switch_model(&working, &working_estimate);
				prob=prob_train ;
				working->estimate_model_viterbi(working_estimate);
				prob_train=working_estimate->best_path(-1);

				if (prob_max<prob_train)
				{
					prob_max=prob_train ;
#ifdef TMP_SAVE
					FILE* file=fopen(templname_best, "w");
					CIO::message(M_INFO, "\nsaving best model with filename %s ... ", templname_best) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message(M_INFO, "done.") ;
#endif
				} 
				else
				{
#ifdef TMP_SAVE
					FILE* file=fopen(templname, "w");
					CIO::message(M_INFO, "\nsaving model with filename %s ... ", templname) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message(M_INFO, "done.") ;
#endif
				}
			}
			delete working_estimate;
			working_estimate=NULL;
		}
		else
			CIO::message(M_ERROR, "assign observation first\n");
	}
	else
		CIO::message(M_ERROR, "create model first\n");

	return false;
}

bool CGUIHMM::viterbi_train_defined(CHAR* param)
{
	CHAR* templname= TMP_DIR "vitdef_model_XXXXXX" ;
#ifdef SUNOS
#define mkstemp(name) mktemp(name);
#endif
	mkstemp(templname);
	CHAR templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
	iteration_count=ITERATIONS ;

	if (working) 
	{
		if (working->get_observations())
		{
			CHMM* working_estimate=new CHMM(working,number_of_hmm_tables);

			double prob_train=CMath::ALMOST_NEG_INFTY, prob = -CMath::INFTY ;

			while (!converge(prob,prob_train))
			{
				switch_model(&working, &working_estimate);
				prob=prob_train ;
				working->estimate_model_viterbi_defined(working_estimate);
				prob_train=working_estimate->best_path(-1);

				if (prob_max<prob_train)
				{
					prob_max=prob_train ;
#ifdef TMP_SAVE
					FILE* file=fopen(templname_best, "w");
					CIO::message(M_INFO, "\nsaving best model with filename %s ... ", templname_best) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message(M_INFO, "done.") ;
#endif
				} 
				else
				{
#ifdef TMP_SAVE
					FILE* file=fopen(templname, "w");
					CIO::message(M_INFO, "\nsaving model with filename %s ... ", templname) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message(M_INFO, "done.") ;
#endif
				}
			}
			delete working_estimate;
			working_estimate=NULL;
		}
		else
			CIO::message(M_ERROR, "assign observation first\n");
	}
	else
		CIO::message(M_ERROR, "create model first\n");

	return false;
}

bool CGUIHMM::linear_train(CHAR* param)
{
	INT numargs=-1;
	CHAR align='l';
	bool right_align=false;

	param=CIO::skip_spaces(param);

	numargs=sscanf(param, "%c", &align);

	if (align=='r')
	{
		CIO::message(M_INFO, "using alignment to right\n");
		right_align=true;
	}
	else
	{
		CIO::message(M_INFO, "using alignment to left\n");
	}

	if ((gui->guifeatures.get_train_features()->get_feature_type() !=F_WORD) ||
			(gui->guifeatures.get_train_features()->get_feature_class() !=C_STRING))
	{
		CIO::message(M_ERROR, "Features must be STRING of type WORD\n");
		return false;
	}

	if (gui->guifeatures.get_train_features())
	{
		working->set_observations((CStringFeatures<WORD>*) gui->guifeatures.get_train_features());
		if (working) 
		{
			working->linear_train(right_align);
			return true;
		}
	}
	else
		CIO::message(M_ERROR, "load train features first\n");

	return false;
}

bool CGUIHMM::one_class_test(CHAR* param)
{
	bool result=false;
	CHAR outputname[1024];
	CHAR rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	INT numargs=-1;
	INT linear=0;

	param=CIO::skip_spaces(param);

	numargs=sscanf(param, "%s %s %d", outputname, rocfname, &linear);

	if (numargs>=1)
	{
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
			CIO::message(M_ERROR, "could not open %s\n",outputname);
			return false;
		}

		if (numargs>=2) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				CIO::message(M_ERROR, "could not open %s\n",rocfname);
				return false;
			}
		}
	}

	if (working)
	{
		if (gui->guifeatures.get_test_features())
		{
			CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) gui->guifeatures.get_test_features();
			CStringFeatures<WORD>* old_test=working->get_observations();

			CLabels* lab=gui->guilabels.get_test_labels();

			working->set_observations(obs);

			INT total=obs->get_num_vectors();
			ASSERT(lab && total == lab->get_num_labels());

			DREAL* output = new DREAL[total];	
			INT* label= new INT[total];	

			for (INT dim=0; dim<total; dim++)
			{
				output[dim]= linear ? working->linear_model_probability(dim) : working->model_probability(dim);
				label[dim]= lab->get_int_label(dim);
			}

			gui->guimath.evaluate_results(output, label, total, outputfile, rocfile);
			delete[] output;
			delete[] label;

			working->set_observations(old_test);

			delete obs;

			result=true;
		}
		else
			CIO::message(M_ERROR, "assign posttest and negtest observations first!\n");
	}
	else
		CIO::message(M_ERROR, "no hmm defined!\n");

	if (rocfile)
		fclose(rocfile);
	if ((outputfile) && (outputfile!=stdout))
		fclose(outputfile);
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
			CIO::message(M_ERROR, "could not open %s\n",outputname);
			return false;
		}

		if (numargs>=2) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				CIO::message(M_ERROR, "could not open %s\n",rocfname);
				return false;
			}
		}
	}

	if (pos && neg)
	{
		if (gui->guifeatures.get_test_features())
		{
			CStringFeatures<WORD>* o= (CStringFeatures<WORD>*) gui->guifeatures.get_test_features();
			CLabels* lab= gui->guilabels.get_test_labels();

			//CStringFeatures<WORD>* old_pos=pos->get_observations();
			//CStringFeatures<WORD>* old_neg=neg->get_observations();

			pos->set_observations(o);
			neg->set_observations(o);

			INT total=o->get_num_vectors();

			DREAL* output = new DREAL[total];	
			INT* label= new INT[total];	

			CIO::message(M_INFO, "classifying using neg %s hmm vs. pos %s hmm\n", neglinear ? "linear" : "", poslinear ? "linear" : "");

			for (INT dim=0; dim<total; dim++)
			{
				output[dim]= 
					(poslinear ? pos->linear_model_probability(dim) : pos->model_probability(dim)) -
					(neglinear ? neg->linear_model_probability(dim) : neg->model_probability(dim));
				label[dim]= lab->get_int_label(dim);
			}

			gui->guimath.evaluate_results(output, label, total, outputfile);

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
		CIO::message(M_ERROR, "assign positive and negative models first!\n");

	if ((outputfile) && (outputfile!=stdout))
		fclose(outputfile);

	return result;
}

bool CGUIHMM::hmm_test(CHAR* param)
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

	numargs=sscanf(param, "%s %s %d %d", outputname, rocfname, &neglinear, &poslinear);

	if (numargs>=1)
	{
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
			CIO::message(M_ERROR, "could not open %s\n",outputname);
			return false;
		}

		if (numargs>=2) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				CIO::message(M_ERROR, "could not open %s\n",rocfname);
				return false;
			}
		}
	}

	if (pos && neg)
	{
		if (gui->guifeatures.get_test_features())
		{
			CStringFeatures<WORD>* o=(CStringFeatures<WORD>*) gui->guifeatures.get_test_features();
			CLabels* lab=gui->guilabels.get_test_labels();

			CStringFeatures<WORD>* old_pos=pos->get_observations();
			CStringFeatures<WORD>* old_neg=neg->get_observations();

			ASSERT(o);
			pos->set_observations(o);
			neg->set_observations(o);

			INT total=o->get_num_vectors();
			ASSERT(lab && total==lab->get_num_labels());

			DREAL* output = new DREAL[total];	
			INT* label= new INT[total];	

			CIO::message(M_INFO, "testing using neg %s hmm vs. pos %s hmm\n", neglinear ? "linear" : "", poslinear ? "linear" : "");

			for (INT dim=0; dim<total; dim++)
			{
				output[dim]= 
					(poslinear ? pos->linear_model_probability(dim) : pos->model_probability(dim)) -
					(neglinear ? neg->linear_model_probability(dim) : neg->model_probability(dim));
				label[dim]= lab->get_int_label(dim);
				//fprintf(outputfile, "%+d: %f - %f = %f\n", label[dim], pos->model_probability(dim), neg->model_probability(dim), output[dim]);
			}

			gui->guimath.evaluate_results(output, label, total, outputfile, rocfile);

			delete[] output;
			delete[] label;

			pos->set_observations(old_pos);
			neg->set_observations(old_neg);

			result=true;
		}
		else
			CIO::message(M_ERROR, "load test features first!\n");
	}
	else
		CIO::message(M_ERROR, "assign positive and negative models first!\n");

	if (rocfile)
		fclose(rocfile);
	if ((outputfile) && (outputfile!=stdout))
		fclose(outputfile);

	return result;
}

CLabels* CGUIHMM::classify(CLabels* result)
{
	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) gui->guifeatures.get_test_features();
	INT num_vec=obs->get_num_vectors();

	if (!result)
		result=new CLabels(num_vec);

	//CStringFeatures<WORD>* old_pos=pos->get_observations();
	//CStringFeatures<WORD>* old_neg=neg->get_observations();

	ASSERT(obs!=NULL);
	pos->set_observations(obs);
	neg->set_observations(obs);

	for (INT i=0; i<num_vec; i++)
		result->set_label(i, pos->model_probability(i) - neg->model_probability(i));

	//pos->set_observations(old_pos);
	//neg->set_observations(old_neg);
	return result;
}

DREAL CGUIHMM::classify_example(INT idx)
{
	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) gui->guifeatures.get_test_features();

	//CStringFeatures<WORD>* old_pos=pos->get_observations();
	//CStringFeatures<WORD>* old_neg=neg->get_observations();

	ASSERT(obs!=NULL);
	pos->set_observations(obs);
	neg->set_observations(obs);

	DREAL result=pos->model_probability(idx) - neg->model_probability(idx);
	//pos->set_observations(old_pos);
	//neg->set_observations(old_neg);
	return result;
}

CLabels* CGUIHMM::one_class_classify(CLabels* result)
{
	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) gui->guifeatures.get_test_features();
	INT num_vec=obs->get_num_vectors();

	if (!result)
	  result=new CLabels(num_vec);

	ASSERT(working);

	//CStringFeatures<WORD>* old_pos=working->get_observations();

	ASSERT(obs!=NULL);
	working->set_observations(obs);


	for (INT i=0; i<num_vec; i++)
		result->set_label(i, working->model_probability(i));

	//working->set_observations(old_pos);
	return result;
}

CLabels* CGUIHMM::linear_one_class_classify(CLabels* result)
{
	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) gui->guifeatures.get_test_features();
	INT num_vec=obs->get_num_vectors();

	if (!result)
		result=new CLabels(num_vec);

	//CStringFeatures<WORD>* old_pos=working->get_observations();

	ASSERT(obs!=NULL);
	working->set_observations(obs);

	ASSERT(working);

	for (INT i=0; i<num_vec; i++)
		result->set_label(i, working->linear_model_probability(i));

	//working->set_observations(old_pos);
	return result;
}


DREAL CGUIHMM::one_class_classify_example(INT idx)
{
	CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) gui->guifeatures.get_test_features();

	//CStringFeatures<WORD>* old_pos=pos->get_observations();

	ASSERT(obs!=NULL);
	pos->set_observations(obs);
	neg->set_observations(obs);

	ASSERT(working);
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

				CHMM* h=new CHMM(model_file,PSEUDO,number_of_hmm_tables);
				if (h && h->get_status())
				{
					printf("file successfully read\n");
					fclose(model_file);

					DREAL* cur_o=new DREAL[h->get_M()];
					DREAL* app_o=new DREAL[h->get_M()];
					ASSERT(cur_o != NULL && app_o != NULL);

					CIO::message(M_DEBUG, "h %d , M: %d\n", h, h->get_M());

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
					CIO::message(M_INFO, "new model has %i states\n", working->get_N());
					delete h;
				}
				else
					CIO::message(M_ERROR, "reading file %s failed\n", fname);
			}
			else
				CIO::message(M_ERROR, "opening file %s failed\n", fname);
		}
		else
			CIO::message(M_ERROR, "see help for parameters\n", fname);
	}
	else
		CIO::message(M_ERROR, "create model first\n");


	return false;
}

bool CGUIHMM::add_states(CHAR* param)
{
	if (working)
	{
		INT states=1;
		double value=0;

		param=CIO::skip_spaces(param);

		sscanf(param, "%i %le", &states, &value);
		CIO::message(M_INFO, "adding %i states\n", states);
		working->add_states(states, value);
		CIO::message(M_INFO, "new model has %i states\n", working->get_N());
		return true;
	}
	else
		CIO::message(M_INFO, "create model first\n");

	return false;
}

bool CGUIHMM::set_pseudo(CHAR* param)
{
	param=CIO::skip_spaces(param);

	if (sscanf(param, "%le", &PSEUDO)!=1)
	{
		CIO::message(M_INFO, "see help for parameters. current setting: pseudo=%e\n", PSEUDO);
		return false ;
	}
	CIO::message(M_INFO, "current setting: pseudo=%e\n", PSEUDO);
	return true ;
}

bool CGUIHMM::convergence_criteria(CHAR* param)
{
	INT j=100;
	double f=0.001;

	param=CIO::skip_spaces(param);

	if (sscanf(param, "%d %le", &j, &f) == 2)
	{
		ITERATIONS=j;
		EPSILON=f;
	}
	else
	{
		CIO::message(M_ERROR, "see help for parameters. current setting: iterations=%i, epsilon=%e\n",ITERATIONS,EPSILON);
		return false ;
	}
	CIO::message(M_INFO, "current setting: iterations=%i, epsilon=%e\n",ITERATIONS,EPSILON);
	return true ;
} ;

bool CGUIHMM::set_hmm_as(CHAR* param)
{
	param=CIO::skip_spaces(param);
	CHAR target[1024];

	if ((sscanf(param, "%s", target))==1)
	{
		if (working)
		{
			if (strcmp(target,"POS")==0)
			{
				delete pos;
				pos=working;
				working=NULL;
			}
			else if (strcmp(target,"NEG")==0)
			{
				delete neg;
				neg=working;
				working=NULL;
			}
			else if (strcmp(target,"TEST")==0)
			{
				delete test;
				test=working;
				working=NULL;
			}
			else
				CIO::message(M_ERROR, "target POS|NEG|TEST missing\n");
		}
		else
			CIO::message(M_ERROR, "create model first!\n");
	}
	else
		CIO::message(M_ERROR, "target POS|NEG|TEST missing\n");

	return false;
}

//convergence criteria  -tobeadjusted-
bool CGUIHMM::converge(double x, double y)
{
	double diff=y-x;
	double absdiff=fabs(diff);

	CIO::message(M_INFO, "\n #%03d\tbest result so far: %G (eps: %f)", iteration_count, y, diff);

	if (iteration_count-- == 0 || (absdiff<EPSILON && conv_it<=0))
	{
		iteration_count=ITERATIONS;
		CIO::message(M_INFO, "...finished\n");
		conv_it=5 ;
		return true;
	}
	else
	{
		if (absdiff<EPSILON)
			conv_it-- ;
		else
			conv_it=5;

		return false;
	}
}

//switch model and train model
void CGUIHMM::switch_model(CHMM** m1, CHMM** m2)
{
	CHMM* dummy= *m1;

	*m1= *m2;
	*m2= dummy;
}

bool CGUIHMM::load(CHAR* param)
{
	bool result=false;

	param=CIO::skip_spaces(param);

	if (working)
		delete working;
	working=NULL;

	FILE* model_file=fopen(param, "r");

	if (model_file)
	{
		working=new CHMM(model_file,PSEUDO,number_of_hmm_tables);
		fclose(model_file);

		if (working && working->get_status())
		{
			printf("file successfully read\n");
			result=true;
		}

		M=working->get_M();
	}
	else
		CIO::message(M_ERROR, "opening file %s failed\n", param);

	return result;
}

bool CGUIHMM::save(CHAR* param)
{
	bool result=false;
	param=CIO::skip_spaces(param);
	CHAR fname[1024];
	INT binary=0;

	if (working)
	{
		if (sscanf(param, "%s %d", fname, &binary) >= 1)
		{
			FILE* file=fopen(fname, "w");
			if (file)
			{
				if (binary)
					result=working->save_model_bin(file);
				else
					result=working->save_model(file);
			}

			if (!file || !result)
				printf("writing to file %s failed!\n", fname);
			else
				printf("successfully written model into \"%s\" !\n", fname);

			if (file)
				fclose(file);
		}
		else
			CIO::message(M_ERROR, "see help for parameters\n");
	}
	else
		CIO::message(M_ERROR, "create model first\n");

	return result;
}

bool CGUIHMM::load_defs(CHAR* param)
{
	param=CIO::skip_spaces(param);
	CHAR fname[1024];
	INT init=1;

	if (working)
	{
		if (sscanf(param, "%s %d", fname, &init) >= 1)
		{
			FILE* def_file=fopen(fname, "r");
			if (def_file && working->load_definitions(def_file,true,(init!=0)))
			{
				CIO::message(M_INFO, "file successfully read\n");
				return true;
			}
			else
				CIO::message(M_ERROR, "opening file %s failed\n", fname);
		}
		else
			CIO::message(M_ERROR, "see help for parameters\n");
	}
	else
		CIO::message(M_ERROR, "create model first\n");

	return false;
}

bool CGUIHMM::save_likelihood(CHAR* param)
{
	bool result=false;
	param=CIO::skip_spaces(param);
	CHAR fname[1024];
	INT binary=0;

	if (working)
	{
		if (sscanf(param, "%s %d", fname, &binary) >= 1)
		{
			FILE* file=fopen(fname, "w");
			if (file)
			{
				/// ..future
				//if (binary)
				//	result=working->save_model_bin(file);
				//else

				result=working->save_likelihood(file);
			}

			if (!file || !result)
				printf("writing to file %s failed!\n", fname);
			else
				printf("successfully written likelihoods into \"%s\" !\n", fname);

			if (file)
				fclose(file);
		}
		else
			CIO::message(M_ERROR, "see help for parameters\n");
	}
	else
		CIO::message(M_ERROR, "create model first\n");

	return result;
}

bool CGUIHMM::save_path(CHAR* param)
{
	bool result=false;
	param=CIO::skip_spaces(param);
	CHAR fname[1024];
	INT binary=0;

	if (working)
	{
	  if (sscanf(param, "%s %d", fname, &binary) >= 1)
	    {
	      FILE* file=fopen(fname, "w");
	      if (file)
		{
		  /// ..future
		  //if (binary)
		  //	result=working->save_model_bin(file);
		  //else
		  CStringFeatures<WORD>* obs= (CStringFeatures<WORD>*) gui->guifeatures.get_test_features();

		  ASSERT(obs!=NULL);
		  working->set_observations(obs);
		  
		  result=working->save_path(file);
		}
	      
	      if (!file || !result)
		printf("writing to file %s failed!\n", fname);
	      else
		printf("successfully written path into \"%s\" !\n", fname);
	      
	      if (file)
		fclose(file);
	    }
	  else
	    CIO::message(M_ERROR, "see help for parameters\n");
	}
	else
	  CIO::message(M_ERROR, "create model first\n");
	
	return result;
}

bool CGUIHMM::chop(CHAR* param)
{
	param=CIO::skip_spaces(param);
	double value;

	if (sscanf(param, "%le", &value) == 1)
	{
		if (working)
			working->chop(value);
		return true;
	}
	else
		CIO::message(M_ERROR, "see help for parameters/create model first\n");
	return false;
}

bool CGUIHMM::likelihood(CHAR* param)
{
	if (working)
	{
		working->output_model(false);
		return true;
	}
	else
		CIO::message(M_ERROR, "create model first!\n");
	return false;
}

bool CGUIHMM::output_hmm(CHAR* param)
{
	if (working)
	{
		working->output_model(true);
		return true;
	}
	else
		CIO::message(M_ERROR, "create model first!\n");
	return false;
}

bool CGUIHMM::output_hmm_defined(CHAR* param)
{
	if (working)
	{
		working->output_model_defined(true);
		return true;
	}
	else
		CIO::message(M_ERROR, "create model first!\n");
	return false;
}


bool CGUIHMM::best_path(CHAR* param)
{
	param=CIO::skip_spaces(param);
	INT from, to;

	if (sscanf(param, "%d %d", &from, &to) != 2)
	{
		from=0; 
		to=100;
	}

	if (working)
	{
		//get path
		working->best_path(0);

		for (INT t=0; t<working->get_observations()->get_vector_length(0)-1 && t<to; t++)
			CIO::message(M_MESSAGEONLY, "%d ", working->get_best_path_state(0,t));
		CIO::message(M_MESSAGEONLY, "\n");
		//for (t=0; t<p_observations->get_vector_length(0)-1 && t<to; t++)
		//	CIO::message(M_MESSAGEONLY, "%d ", PATH(0)[t]);
		return true;
	}
	else
		CIO::message(M_ERROR, "create model first\n");

	return false;
}

bool CGUIHMM::normalize(CHAR* param)
{
	param=CIO::skip_spaces(param);
	INT keep_dead_states=0;
	sscanf(param, "%d", &keep_dead_states);

	if (working)
	{
		working->normalize(keep_dead_states==1);
		return true;
	}
	else
		CIO::message(M_ERROR, "create model first\n");

	return false;
}

bool CGUIHMM::relative_entropy(CHAR* param)
{
	if (pos && neg) 
	{
		if ( (pos->get_M() == neg->get_M()) && (pos->get_N() == neg->get_N()) )
		{
			double* _entropy=new double[pos->get_N()];
			double* p=new double[pos->get_M()];
			double* q=new double[pos->get_M()];

			for (INT i=0; i<pos->get_N(); i++)
			{
				for (INT j=0; j<pos->get_M(); j++)
				{
					p[j]=pos->get_b(i,j);
					q[j]=neg->get_b(i,j);
				}

				_entropy[i]=CMath::relative_entropy(p, q, pos->get_M());
				CIO::message(M_MESSAGEONLY, "%f ", _entropy[i]);
			}
			CIO::message(M_MESSAGEONLY, "\n");
			delete[] p;
			delete[] q;
			delete[] _entropy;
		}
		else
			CIO::message(M_ERROR, "pos and neg hmm's differ in number of emissions or states\n");
	}
	else
		CIO::message(M_ERROR, "set pos and neg hmm first\n");
	return false;
}

bool CGUIHMM::entropy(CHAR* param)
{
	if (pos) 
	{
		double* _entropy=new double[pos->get_N()];
		double* p=new double[pos->get_M()];

		for (INT i=0; i<pos->get_N(); i++)
		{
			for (INT j=0; j<pos->get_M(); j++)
			{
				p[j]=pos->get_b(i,j);
			}

			_entropy[i]=CMath::entropy(p, pos->get_M());
			CIO::message(M_MESSAGEONLY, "%f ", _entropy[i]);
		}
		CIO::message(M_MESSAGEONLY, "\n");

		delete[] p;
		delete[] _entropy;
	}
	else
		CIO::message(M_ERROR, "set pos hmm first\n");
	return false;
}

bool CGUIHMM::permutation_entropy(CHAR* param)
{
	param=CIO::skip_spaces(param);

	INT width=0;
	INT seq_num=-1;

	if (sscanf(param, "%d %d", &width, &seq_num) == 2)
	{
		if (working) 
		{
			if (working->get_observations())
			{
				return working->permutation_entropy(width, seq_num);
			}
			else
				CIO::message(M_ERROR, "set observations first\n");
		}
		else
			CIO::message(M_ERROR, "create hmm first\n");
	}
	else
		CIO::message(M_ERROR, "wrong number of parameters see help!\n");

	return false;
}
#endif
