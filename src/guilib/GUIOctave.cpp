/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#if defined(HAVE_OCTAVE) && !defined(HAVE_SWIG)
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <sstream>
#include <string>

using namespace std;

#include "guilib/GUIOctave.h"
#include "guilib/GUIMatlab.h"
#include "gui/TextGUI.h"
#include "gui/GUI.h"

#include "lib/io.h"
#include "distributions/hmm/HMM.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "features/CharFeatures.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "classifier/svm/SVM.h"

extern CTextGUI* gui;

CGUIOctave::CGUIOctave() : CSGObject()
{
}

bool CGUIOctave::send_command(CHAR* cmd)
{
	return (gui->parse_line(cmd));
}

bool CGUIOctave::get_hmm(octave_value_list& retvals)
{
	CHMM* h=gui->guihmm.get_current();

	if (h)
	{
		RowVector vec_p=RowVector(h->get_N());
		RowVector vec_q=RowVector(h->get_N());
		Matrix mat_a=Matrix(h->get_N(), h->get_N());
		Matrix mat_b=Matrix(h->get_N(), h->get_M());

		if ((vec_p.length() == h->get_N()) && 
				(vec_q.length() == h->get_N()) &&
				(mat_a.rows() == h->get_N()) && (mat_a.cols() == h->get_N()) &&
				(mat_b.rows() == h->get_N()) && (mat_b.cols() == h->get_M()))
		{
			int i,j;
			for (i=0; i< h->get_N(); i++)
			{
				vec_p(i)=h->get_p(i);
				vec_q(i)=h->get_q(i);
			}

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_N(); j++)
					mat_a(i,j)=h->get_a(i,j);

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_M(); j++)
					mat_b(i,j)=h->get_b(i,j);

			retvals(0)=vec_p;
			retvals(1)=vec_q;
			retvals(2)=mat_a;
			retvals(3)=mat_b;

			return true;
		}
		else
			SG_ERROR( "creating vectors/matrices failed\n");
	}

	return false;
}

bool CGUIOctave::hmm_likelihood(octave_value_list& retvals)
{
	CHMM* h=gui->guihmm.get_current();

	if (h)
	{

		RowVector s=RowVector(1);
		s(0)=h->model_probability();
		retvals(0)=s;
		return true;
	}
	return false;
}

bool CGUIOctave::best_path(octave_value_list& retvals, int dim)
{
	CHMM* h=gui->guihmm.get_current();
	SG_DEBUG( "dim: %d\n", dim);

	if (h)
	{
		CFeatures* f=gui->guifeatures.get_test_features();

		if ((f) && (f->get_feature_class() == C_STRING)
				&& (f->get_feature_type() == F_WORD)
		   )
		{
			h->set_observations((CStringFeatures<WORD>*) f);
			INT num_feat;

			WORD* fv = ((CStringFeatures<WORD>*) f)->get_feature_vector(dim, num_feat);

			SG_DEBUG( "computing viterbi path for vector %d (length %d)\n", dim, num_feat);

			if (fv && num_feat>0)
			{
				RowVector path = RowVector(num_feat);
				RowVector lik = RowVector(1);

				double l=0;
				T_STATES* s = h->get_path(dim, l);
				lik(0)=l;

				for (int i=0; i<num_feat; i++)
					path(i)=s[i];

				delete[] s;

				retvals(0)=path;
				retvals(1)=lik;
				return true;
			}
		}
	}
	return false;
}

bool CGUIOctave::append_hmm(const octave_value_list& vals)
{
	CHMM* old_h=gui->guihmm.get_current();
	ASSERT(old_h);

	const RowVector vec_p=vals(1).row_vector_value();
	const RowVector vec_q=vals(2).row_vector_value();
	const Matrix mat_a=vals(3).matrix_value();
	const Matrix mat_b=vals(4).matrix_value();

	INT N=mat_a.cols();
	INT M=mat_b.cols();

	if ( N > 0 && 
			M > 0 &&
			vec_p.length() == N &&
			vec_q.length() == N &&
			mat_a.rows() == N &&
			mat_b.rows() == N
	   )
	{
		CHMM* h=new CHMM(N, M, NULL, gui->guihmm.get_pseudo());
		if (h)
		{
			SG_DEBUG( "N:%d M:%d p:(%d) q:(%d) a:(%d,%d) b(%d,%d)\n",
					N, M,
					vec_p.length(),
					vec_q.length(),
					mat_a.rows(), mat_a.cols(), 
					mat_b.rows(), mat_b.cols());

			int i,j;
			for (i=0; i< h->get_N(); i++)
			{
				h->set_p(i, vec_p(i));
				h->set_q(i, vec_q(i));
			}

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_N(); j++)
					h->set_a(i,j, mat_a(i,j));

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_M(); j++)
					h->set_b(i,j, mat_b(i,j));

			SG_INFO( "h %d , M: %d\n", h, h->get_M());

			old_h->append_model(h);

			delete h;

			return true;
		}
		else
			SG_ERROR( "creating vectors/matrices failed\n");
	}
	else
		SG_ERROR( "model matricies not matching in size\n");
	return false;
}

bool CGUIOctave::set_hmm(const octave_value_list& vals)
{
	const RowVector vec_p=vals(1).row_vector_value();
	const RowVector vec_q=vals(2).row_vector_value();
	const Matrix mat_a=vals(3).matrix_value();
	const Matrix mat_b=vals(4).matrix_value();

	INT N=mat_a.cols();
	INT M=mat_b.cols();

	CHMM* h=new CHMM(N, M, NULL, gui->guihmm.get_pseudo());

	if ( N > 0 && 
			M > 0 &&
			vec_p.length() == N &&
			vec_q.length() == N &&
			mat_a.rows() == N &&
			mat_b.rows() == N
	   )
	{
		if (h)
		{
			SG_DEBUG( "N:%d M:%d p:(%d) q:(%d) a:(%d,%d) b(%d,%d)\n",
					N, M,
					vec_p.length(),
					vec_q.length(),
					mat_a.rows(), mat_a.cols(), 
					mat_b.rows(), mat_b.cols());

			int i,j;
			for (i=0; i< h->get_N(); i++)
			{
				h->set_p(i, vec_p(i));
				h->set_q(i, vec_q(i));
			}

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_N(); j++)
					h->set_a(i,j, mat_a(i,j));

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_M(); j++)
					h->set_b(i,j, mat_b(i,j));

			gui->guihmm.set_current(h);
			return true;
		}
	}
	else
		SG_ERROR( "model matricies not matching in size\n");

	return false;
}

bool CGUIOctave::hmm_classify(octave_value_list& retvals)
{
	CFeatures* f=gui->guifeatures.get_test_features();

	if (f)
	{
		int num_vec = f->get_num_vectors();

		RowVector result = RowVector(num_vec);
		CLabels* l=gui->guihmm.classify();

		for (int i=0; i<num_vec; i++)
			result(i)=l->get_label(i);

		delete l;

		retvals(0) = result;
		return true;
	}
	return false;
}

bool CGUIOctave::hmm_classify_example(octave_value_list& retvals, int idx)
{
	double result=gui->guihmm.classify_example(idx);
	RowVector r=RowVector(1);
	r(0)=result;
	retvals(0) = r;
	return true;
}

bool CGUIOctave::one_class_hmm_classify(octave_value_list& retvals, bool linear)
{
	CFeatures* f=gui->guifeatures.get_test_features();
	if (f)
	{
		int num_vec=f->get_num_vectors();

		ASSERT(num_vec);

		RowVector result = RowVector(num_vec);

		CLabels* l         = NULL ;
		if (!linear)
			l=gui->guihmm.one_class_classify();
		else
			l=gui->guihmm.linear_one_class_classify();

		for (int i=0; i<num_vec; i++)
			result(i)=l->get_label(i);

		delete l;

		retvals(0)=result;
		return true;
	}
	return false;
}

bool CGUIOctave::one_class_hmm_classify_example(octave_value_list& retvals, int idx)
{
	retvals(0)= gui->guihmm.one_class_classify_example(idx);
	return true;
}

bool CGUIOctave::get_svm(octave_value_list& retvals)
{
	CSVM* svm=gui->guisvm.get_svm();

	if (svm)
	{
		Matrix alphas=Matrix(svm->get_num_support_vectors(), 2);
		double b=0;

		if (alphas.rows() == svm->get_num_support_vectors() &&
				alphas.cols() == 2)
		{
			b=svm->get_bias();

			for (int i=0; i< svm->get_num_support_vectors(); i++)
			{
				alphas(1,i)=svm->get_alpha(i);
				alphas(2,i)=svm->get_support_vector(i);
			}

			retvals(0)=b;
			retvals(1)=alphas;

			return true;
		}
	}

	return false;
}

bool CGUIOctave::set_svm(const octave_value_list& vals)
{
	CSVM* svm=gui->guisvm.get_svm();

	if (svm)
	{
		double b=vals(1).double_value();
		Matrix alphas=vals(2).matrix_value();

		if ( alphas.cols() == 2 )
		{
			svm->create_new_model(alphas.rows());
			svm->set_bias(b);

			for (int i=0; i< svm->get_num_support_vectors(); i++)
			{
				svm->set_alpha(i, alphas(1, i));
				svm->set_support_vector(i, (int) alphas(2,i));
			}

			return true;
		}
	}

	return false;
}

bool CGUIOctave::svm_classify(octave_value_list& retvals)
{
	CFeatures* f=gui->guifeatures.get_test_features();
	if (f)
	{
		int num_vec=f->get_num_vectors();

		CLabels* l=gui->guisvm.classify();

		if (!l)
		{
			SG_ERROR( "svm_classify failed\n") ;
			return false ;
		} ;

		RowVector result=RowVector(num_vec);
		for (int i=0; i<num_vec; i++)
			result(i)=l->get_label(i);
		delete l;

		retvals(0)=result;
		return true;
	}
	return false;
}

bool CGUIOctave::svm_classify_example(octave_value_list& retvals, int idx)
{
	double result=0;

	if (!gui->guisvm.classify_example(idx, result))
	{
		SG_ERROR( "svm_classify_example failed\n") ;
		return false ;
	}

	RowVector r=RowVector(1);
	r(0)=result;
	retvals(0)=r;

	return true;
}

bool CGUIOctave::set_plugin_estimate(const octave_value_list& vals)
{
	Matrix model_parm = vals(1).matrix_value();
	RowVector sizes = vals(2).row_vector_value();

	int num_params = model_parm.rows();
	ASSERT(model_parm.cols()==2);

	const double* result=model_parm.data();
	const DREAL* pos_params = result;
	const DREAL* neg_params = &(result[num_params]) ;

	int seq_length = (int) sizes(0);
	int num_symbols = (int) sizes(1);
	ASSERT(num_params == seq_length*num_symbols) ;

	gui->guipluginestimate.get_estimator()->set_model_params(pos_params, neg_params, seq_length, num_symbols) ;

	return true;
}

bool CGUIOctave::get_plugin_estimate(octave_value_list& retvals)
{
	DREAL* pos_params, * neg_params ;
	int num_params = 0, seq_length=0, num_symbols=0 ;

	if (!gui->guipluginestimate.get_estimator()->get_model_params(pos_params, neg_params, seq_length, num_symbols))
		return false ;

	num_params = seq_length * num_symbols ;
	Matrix result=Matrix(num_params, 2);

	for (int i=0; i<num_params; i++)
		result(0,i) = pos_params[i] ;
	for (int i=0; i<num_params; i++)
		result(1,i) = neg_params[i] ;

	retvals(0) = result;

	RowVector sizes = RowVector(2);
	sizes(0) = (double) seq_length;
	sizes(1) = (double) num_symbols;

	retvals(1)=sizes ;
	return true;
}

bool CGUIOctave::plugin_estimate_classify(octave_value_list& retvals)
{
	CFeatures* f=gui->guifeatures.get_test_features();
	if (f)
	{
		int num_vec = f->get_num_vectors();

		RowVector result = RowVector(num_vec);
		CLabels* l = gui->guipluginestimate.classify();

		for (int i=0; i<num_vec; i++)
			result(i) = l->get_label(i);

		delete l;

		retvals(0) = result;
		return true;
	}
	return false;
}

bool CGUIOctave::plugin_estimate_classify_example(octave_value_list& retvals, int idx)
{
	retvals(0) = gui->guipluginestimate.classify_example(idx);
	return true;
}

bool CGUIOctave::get_features(octave_value_list& retvals, CFeatures* f)
{
	if (f)
	{
		///octave can only deal with Simple (==rectangular) features

		if (f->get_feature_class()==C_SIMPLE || f->get_feature_class()==C_SPARSE)
		{
			octave_value value;

			switch (f->get_feature_class())
			{
				case C_SIMPLE:
					switch (f->get_feature_type())
					{
						case F_DREAL:
							{
								Matrix mat_feat=Matrix(((CRealFeatures*) f)->get_num_features(), ((CRealFeatures*) f)->get_num_vectors());

								SG_DEBUG( "cols:%d rows:%d\n", mat_feat.cols(), mat_feat.rows());
								if ( (mat_feat.cols() == ((CRealFeatures*) f)->get_num_vectors()) &&
										(mat_feat.rows() == ((CRealFeatures*) f)->get_num_features()) 
								   )
								{
									SG_DEBUG( "conversion\n");
									for (INT i=0; i<((CRealFeatures*) f)->get_num_vectors(); i++)
									{
										INT num_feat;
										bool free_vec;
										DREAL* vec=((CRealFeatures*) f)->get_feature_vector(i, num_feat, free_vec);
										for (INT j=0; j<num_feat; j++)
											mat_feat(j,i) = (double) vec[j];
										((CRealFeatures*) f)->free_feature_vector(vec, i, free_vec);
									}

									value = mat_feat;
								}
							}
							break;
						case F_SHORT:
						case F_CHAR:
						case F_BYTE:
						default:
							io.not_implemented();
					}
					break;
				case C_SPARSE:
					switch (f->get_feature_type())
					{
						case F_DREAL:
						default:
							io.not_implemented();
					};
					break;
				default:
					io.not_implemented();
			}
			if (!value.is_empty())
				retvals(0)=value;

			return (!value.is_empty());
		}
		else
			SG_ERROR( "matlab does not support that feature type\n");

	}

	return false;
}

CFeatures* CGUIOctave::set_features(const octave_value_list& vals)
{
	octave_value mat_feat = vals(2);
	CFeatures* f=NULL;

	if (!mat_feat.is_empty())
	{
		if (mat_feat.is_cell())
		{
			Cell c=mat_feat.cell_value();
			SG_DEBUG( "cell has %d cols, %d rows\n", c.cols(), c.rows());
			ASSERT(c.rows() == 1 && c(0,0).is_char_matrix());

			int num_vec=c.cols();
			ASSERT(num_vec>=1);


			if (c(0,0).char_matrix_value()(0,0))
			{
				if (vals.length()==4)
				{
					CHAR* al = CGUIOctave::get_octaveString(vals(3).string_value());
					CAlphabet* alpha = new CAlphabet(al, strlen(al));
					T_STRING<CHAR>* sc=new T_STRING<CHAR>[num_vec];
					ASSERT(alpha);
					ASSERT(sc);

					int maxlen=0;
					alpha->clear_histogram();

					for (int i=0; i<num_vec; i++)
					{
						//note the .string here is 0 terminated although it is not required
						charMatrix line=c(0,i).char_matrix_value();
						ASSERT(line.rows() == 1);
						//.length is the length of the string w/o 0
						sc[i].string=new CHAR[line.length()];

						if (sc[i].string)
						{
							for (int j=0; j<line.length(); j++)
								sc[i].string[j]=line(0,j);
							sc[i].length=line.length(); 
							maxlen=CMath::max(maxlen, sc[i].length);
							alpha->add_string_to_histogram(sc[i].string, sc[i].length);
						}
						else
						{
							SG_WARNING( "string with index %d has zero length\n", i+1);
							sc[i].length=0;
						}
					}

                    SG_INFO("max_value_in_histogram:%d\n", alpha->get_max_value_in_histogram());
                    SG_INFO("num_symbols_in_histogram:%d\n", alpha->get_num_symbols_in_histogram());
					f= new CStringFeatures<CHAR>(alpha);
					ASSERT(f);

					if (alpha->check_alphabet_size() && alpha->check_alphabet())
						((CStringFeatures<CHAR>*) f)->set_features(sc, num_vec, maxlen);
					else
					{
						((CStringFeatures<CHAR>*) f)->set_features(sc, num_vec, maxlen);
						delete f;
						f=NULL;
					}
				}
				else
					SG_ERROR( "please specify alphabet!\n");
			}
		}
		else
		{
			if (mat_feat.is_char_matrix())
			{
				charMatrix cm = mat_feat.char_matrix_value();

				if (vals.length()==4)
				{
					CHAR* al=CGUIOctave::get_octaveString(vals(3).string_value());
					CAlphabet* alpha= new CAlphabet(al, strlen(al));
					
					INT num_vec = cm.cols();
					INT num_feat = cm.rows();
					
					CHAR* fm=new CHAR[num_vec*num_feat];
					ASSERT(fm);

					for (INT i=0; i<num_vec; i++)
						for (INT j=0; j<num_feat; j++)
							fm[i*num_feat+j]= (CHAR) cm(j,i);

					alpha->add_string_to_histogram(fm, ((LONG) num_vec)* ((LONG) num_feat));
                    SG_INFO("max_value_in_histogram:%d\n", alpha->get_max_value_in_histogram());
                    SG_INFO("num_symbols_in_histogram:%d\n", alpha->get_num_symbols_in_histogram());

					f= new CCharFeatures(alpha, 0);
					ASSERT(f);

					if (alpha->check_alphabet_size() && alpha->check_alphabet())
						((CCharFeatures*) f)->set_feature_matrix(fm, num_feat, num_vec);
					else
					{
						((CCharFeatures*) f)->set_feature_matrix(fm, num_feat, num_vec);
						delete f;
						f=NULL;
					}
				}
				else
					SG_ERROR( "please specify alphabet!\n");

			}
			else if (mat_feat.is_real_matrix())
			{
				Matrix m = mat_feat.matrix_value();

				f = new CRealFeatures(0);
				INT num_vec = m.cols();
				INT num_feat = m.rows();
				SG_DEBUG( "vec: %d feat:%d\n", num_vec, num_feat);
				DREAL* fm = new DREAL[num_vec*num_feat];
				ASSERT(fm);

				for (INT i=0; i<num_vec; i++)
					for (INT j=0; j<num_feat; j++)
						fm[i*num_feat+j]= (double) m(j,i);

				((CRealFeatures*) f)->set_feature_matrix(fm, num_feat, num_vec);
			}
			///and so on
			else
				SG_ERROR( "not implemented\n");
		}
	}
	return f;
}

bool CGUIOctave::get_labels(octave_value_list& retvals, CLabels* label)
{
	if (label)
	{
		RowVector lab = RowVector(label->get_num_labels());

		if (lab.cols())
		{
			for (int i=0; i< label->get_num_labels(); i++)
				lab(i) = label->get_label(i);

			retvals(0) = lab;
			return true;
		}
	}

	return false;
}

CLabels* CGUIOctave::set_labels(const octave_value_list& vals)
{
	octave_value mat_feat = vals(2);
	Matrix m = mat_feat.matrix_value();
	INT num= m.cols();
	SG_DEBUG( "num: %d\n",num);

	CLabels* label=new CLabels(num);

	SG_INFO( "%d\n", label->get_num_labels());

	for (int i=0; i<label->get_num_labels(); i++)
		label->set_label(i, m(0,i));

	return label;
}


CHAR* CGUIOctave::get_octaveString(std::string s)
{
	CHAR* cstr = strdup(s.c_str());
	return cstr;
}

bool CGUIOctave::get_kernel_matrix(octave_value_list& retvals)
{
	CKernel *k = gui->guikernel.get_kernel();

	if (k && k->get_rhs() && k->get_lhs())
	{
		int num_vec1=k->get_lhs()->get_num_vectors();
		int num_vec2=k->get_rhs()->get_num_vectors();

		Matrix result = Matrix(num_vec1, num_vec2);

		for (int i=0; i<num_vec1; i++)
			for (int j=0; j<num_vec2; j++)
				result(i,j) = k->kernel(i,j);

		retvals(0) = result;
		return true;
	}
	return false;
}

bool CGUIOctave::get_kernel_optimization(octave_value_list& retvals)
{
	CKernel *kernel_ = gui->guikernel.get_kernel() ;

	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
	{
		CWeightedDegreeCharKernel *kernel = (CWeightedDegreeCharKernel *) kernel_ ;

		if (kernel->get_max_mismatch()!=0)
			return false;

		INT len=0 ;
		DREAL* res = kernel->compute_abs_weights(len) ;

		Matrix result=Matrix(4, len);
		for (int i=0; i<4*len; i++)
			result(i) = res[i];

		delete[] res;

		retvals(0) = result;
		return true;
	}

	//if (kernel_ && (kernel_->get_kernel_type() == K_COMMWORDSTRING))
	//{
	//	CCommWordStringKernel *kernel = (CCommWordStringKernel *) kernel_ ;

	//	INT len=0 ;
	//	WORD* dict ;
	//	DREAL* weights ;
	//	//FIXMEkernel->get_dictionary(len, dict, weights) ;

	//	Matrix result = Matrix(len, 2);

	//	for (int i=0; i<len; i++)
	//		result(0,i) = dict[i] ;
	//	for (int i=0; i<len; i++)
	//		result(1,i) = weights[i] ;

	//	retvals(0)=result;
	//	return true;
	//}
	return false;
}
#endif
