/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef HAVE_OCTAVE
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
#include "distributions/hmm/penalty_info.h"
#include "distributions/hmm/HMM.h"
#include "distributions/hmm/penalty_info.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "features/CharFeatures.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "classifier/svm/SVM.h"

extern CTextGUI* gui;

CGUIOctave::CGUIOctave()
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
		fprintf(stderr, "N=%i  M=%i\n", h->get_N(), h->get_M()) ;
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
			CIO::message(M_ERROR, "creating vectors/matrices failed\n");
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
		CHMM* h=new CHMM(N, M, NULL,
				gui->guihmm.get_pseudo(), gui->guihmm.get_number_of_tables());
		if (h)
		{
			CIO::message(M_DEBUG, "N:%d M:%d p:(%d) q:(%d) a:(%d,%d) b(%d,%d)\n",
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

			CIO::message(M_INFO, "h %d , M: %d\n", h, h->get_M());

			old_h->append_model(h);

			delete h;

			return true;
		}
		else
			CIO::message(M_ERROR, "creating vectors/matrices failed\n");
	}
	else
		CIO::message(M_ERROR, "model matricies not matching in size\n");
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

	CHMM* h=new CHMM(N, M, NULL,
			gui->guihmm.get_pseudo(), gui->guihmm.get_number_of_tables());

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
			CIO::message(M_DEBUG, "N:%d M:%d p:(%d) q:(%d) a:(%d,%d) b(%d,%d)\n",
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
		CIO::message(M_ERROR, "model matricies not matching in size\n");

	return false;
}


bool CGUIOctave::best_path_no_b(const octave_value_list& vals, octave_value_list& retvals)
{
	const RowVector vec_p=vals(1).row_vector_value();
	const RowVector vec_q=vals(2).row_vector_value();
	const Matrix mat_a=vals(3).matrix_value();
	INT max_iter=vals(4).int_value();

	INT N=mat_a.cols();

	if ( N > 0 && 
			vec_p.length() == N &&
			vec_q.length() == N &&
			mat_a.rows() == N
	   )
	{
		double* p = (double*) vec_p.fortran_vec();
		double* q = (double*) vec_q.fortran_vec();
		double* a = (double*) mat_a.fortran_vec();

		CHMM* h=new CHMM(N, p, q, a);

		ASSERT(h);

		INT* my_path = new INT[max_iter];
		int best_iter = 0;

		retvals(0) = h->best_path_no_b(max_iter, best_iter, my_path);

		RowVector result = RowVector(best_iter+1);

		for (INT i=0; i<best_iter+1; i++)
			result(i) = my_path[i];

		retvals(1) = result;

		delete h;
		return true;
	}
	else
		CIO::message(M_ERROR, "model matricies not matching in size\n");

	return false;
}

bool CGUIOctave::best_path_no_b_trans(const octave_value_list& vals, octave_value_list& retvals) 
{
	const RowVector vec_p=vals(1).row_vector_value();
	const RowVector vec_q=vals(2).row_vector_value();
	const Matrix mat_a=vals(3).matrix_value();
	INT max_iter=vals(4).int_value();
	INT nbest=vals(5).int_value();

	if (nbest<1)
		return false ;
	if (max_iter<1)
		return false ;

	INT N=vec_p.length();

	if ( N > 0 && 
			vec_p.length() == N &&
			vec_q.length() == N &&
			mat_a.rows() == N
	   )
	{
		double* p = (double*) vec_p.fortran_vec();
		double* q = (double*) vec_q.fortran_vec();
		double* a = (double*) mat_a.fortran_vec();

		CHMM* h = new CHMM(N, p, q, mat_a.rows(), a);

		INT *my_path = new INT[(max_iter+1)*nbest] ;
		memset(my_path, -1, (max_iter+1)*nbest*sizeof(INT)) ;

		int max_best_iter = 0 ;
		RowVector prob = RowVector(nbest);
		DREAL* p_prob = new DREAL[nbest];

		h->best_path_no_b_trans(max_iter, max_best_iter, nbest, p_prob, my_path);

		for (INT i=0; i<nbest; i++)
			prob(i) = p_prob[i];

		delete[] p_prob;

		Matrix result = Matrix(nbest, max_best_iter+1);

		for (INT k=0; k<nbest; k++)
		{
			for (INT i=0; i<max_best_iter+1; i++)
				result(k,i) = my_path[i+k*(max_iter+1)];
		}

		retvals(0)=prob;
		retvals(1)=result;

		delete h;
		delete[] my_path;
		return true;
	}
	else
		CIO::message(M_ERROR, "model matricies not matching in size\n");

	return false;
}


bool CGUIOctave::best_path_trans(const octave_value_list& vals, octave_value_list& retvals)
{
	const RowVector vec_p=vals(1).row_vector_value();
	const RowVector vec_q=vals(2).row_vector_value();
	const Matrix mat_a=vals(3).matrix_value();
	const Matrix mat_seq=vals(4).matrix_value();
	const RowVector vec_pos=vals(5).row_vector_value();
	const Matrix mat_orf_info=vals(6).matrix_value();
	CHAR* genestr = get_octaveString(vals(7).string_value());				
	const RowVector vec_genestr=vals(7).row_vector_value();
	const Matrix mat_penalties=vals(8).matrix_value();
	const Cell cel_penalty_info=vals(9).cell_value();
	INT nbest=vals(10).int_value();
	const Matrix mat_dict_weights=vals(11).matrix_value();

	//const mxArray* mx_p=vals[1];
	//const mxArray* mx_q=vals[2];
	//const mxArray* mx_a_trans=vals[3];
	//const mxArray* mx_seq=vals[4];
	//const mxArray* mx_pos=vals[5];
	//const mxArray* mx_orf_info=vals[6];
	//const mxArray* mx_genestr=vals[7];
	//const mxArray* mx_penalties=vals[8];
	//const mxArray* mx_penalty_info=vals[9];
	//const mxArray* mx_nbest=vals[10];
	//const mxArray* mx_dict_weights=vals[11];

	if (nbest<1)
		return false;

	INT N = vec_p.cols();
	INT M = vec_pos.cols();
	INT P = cel_penalty_info.cols();
	INT L = vec_genestr.cols();
	INT D = mat_dict_weights.rows();

	if (
			vec_p.cols() == N &&
			vec_q.cols() == N &&
			mat_a.cols() == 3 &&
			mat_seq.cols() == N &&
			mat_seq.cols() == vec_pos.cols() &&
			mat_penalties.cols() == N &&
			mat_penalties.rows() == N &&
			mat_orf_info.rows() == N &&
			mat_orf_info.cols() == 2 &&
			mat_dict_weights.cols() == 4 &&
			((vals(9).is_cell() && cel_penalty_info.rows() == 1) ||
			 cel_penalty_info.is_empty())
	   )
	{
		double* p = (double*) vec_p.fortran_vec();
		double* q = (double*) vec_q.fortran_vec();
		double* a = (double*) mat_a.fortran_vec();

		const double* seq = mat_seq.data();

		INT* pos = new INT[M] ;
		INT* orf_info = new INT[2*N];

		for (INT i=0; i<M; i++)
			pos[i] = (INT) vec_pos(i);

		for (INT i=0; i<2*N; i++)
			orf_info[i] = (INT) mat_orf_info(i);

		struct penalty_struct* PEN = NULL;
			////FIXME!!! read_penalty_struct_from_cell(mx_penalty_info, P);
		if (PEN==NULL && P!=0)
			return false ;

		struct penalty_struct** PEN_matrix = new struct penalty_struct*[N*N];
		const double* penalties = mat_penalties.data();
		for (INT i=0; i<N*N; i++)
		{
			INT id = (INT) penalties[i]-1;
			if ((id<0 || id>=P) && (id!=-1))
			{
				CIO::message(M_ERROR, "id out of range\n");
				delete_penalty_struct_array(PEN, P);
				return false ;
			}
			if (id==-1)
				PEN_matrix[i]=NULL;
			else
				PEN_matrix[i]=&PEN[id] ;
		}

		DREAL* dict_weights =  new DREAL[mat_dict_weights.length()];

		CHMM* h=new CHMM(N, p, q, mat_a.rows(), a);

		INT* my_path = new INT[M*nbest];
		memset(my_path, -1, M*nbest*sizeof(INT));
		INT* my_pos = new INT[M*nbest];
		memset(my_pos, -1, M*nbest*sizeof(INT));

		RowVector mat_prob = RowVector(nbest);
		DREAL* p_prob = new DREAL[nbest];

		//FIXME h->best_path_trans(seq, M, pos, orf_info, PEN_matrix, genestr, L,
				//nbest, p_prob, my_path, my_pos, dict_weights, 2*D) ;

		for (int i=0; i<nbest; i++)
			mat_prob(i) = p_prob[i];

		// clean up 
		delete_penalty_struct_array(PEN, P);
		delete[] PEN_matrix;
		delete[] pos;
		delete[] dict_weights;
		delete[] orf_info;
		delete[] p_prob;
		delete h;
		free(genestr);

		// transcribe result
		Matrix mat_my_path = Matrix(nbest, M);
		Matrix mat_my_pos = Matrix(nbest, M);

		for (INT k=0; k<nbest; k++)
			for (INT i=0; i<M; i++)
			{
				mat_my_path(k,i) = my_path[i+k*M];
				mat_my_pos(k,i) = my_pos[i+k*M];
			}

		retvals(0)=mat_prob;
		retvals(1)=mat_my_path;
		retvals(2)=mat_my_pos;
		delete[] my_path;
		delete[] my_pos;

		return true;
	}
	else
		CIO::message(M_ERROR, "model matricies not matching in size\n");
	return false;
}


bool CGUIOctave::model_prob_no_b_trans(const octave_value_list& vals, octave_value_list& retvals)
{
	const RowVector vec_p=vals(1).row_vector_value();
	const RowVector vec_q=vals(2).row_vector_value();
	const Matrix mat_a=vals(3).matrix_value();
	INT max_iter=vals(4).int_value();

	if (max_iter<1)
		return false ;

	INT N=mat_a.cols();

	if ( N > 0 && 
			vec_p.length() == N &&
			vec_q.length() == N &&
			mat_a.rows() == N
	   )
	{
		double* p = (double*) vec_p.fortran_vec();
		double* q = (double*) vec_q.fortran_vec();
		double* a = (double*) mat_a.fortran_vec();

		CHMM* h=new CHMM(N, p, q, mat_a.rows(), a);

		RowVector prob = RowVector(max_iter);
		double* p_prob = new double[max_iter];

		h->model_prob_no_b_trans(max_iter, p_prob);

		for (INT i=0; i<max_iter; i++)
			prob(i) = p_prob[i];

		delete[] p_prob;

		retvals(0) = prob;

		delete h;
		return true;
	}
	else
		CIO::message(M_ERROR, "model matricies not matching in size\n");

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
	retvals(0) = gui->guihmm.classify_example(idx);
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
			CIO::message(M_ERROR, "svm_classify failed\n") ;
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

bool svm_classify_example(octave_value_list& retvals, int idx)
{
	double result=0;

	if (!gui->guisvm.classify_example(idx, result))
	{
		CIO::message(M_ERROR, "svm_classify_example failed\n") ;
		return false ;
	}

	retvals(0)=result;

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
		///matlab can only deal with Simple (==rectangular) features
		///or sparse ones

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
								Matrix mat_feat=Matrix(((CRealFeatures*) f)->get_num_vectors(), ((CRealFeatures*) f)->get_num_features());

								if ( (!mat_feat.cols() == ((CRealFeatures*) f)->get_num_vectors()) &&
										(mat_feat.rows() == ((CRealFeatures*) f)->get_num_features()) 
								   )
								{
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
							CIO::message(M_ERROR, "not implemented\n");
					}
					break;
				case C_SPARSE:
					switch (f->get_feature_type())
					{
						case F_DREAL:
						default:
							CIO::message(M_ERROR, "not implemented\n");
					};
					break;
				default:
					CIO::message(M_ERROR, "not implemented\n");
			}
			if (!value.is_empty())
				retvals(0)=value;

			return (!value.is_empty());
		}
		else
			CIO::message(M_ERROR, "matlab does not support that feature type\n");

	}

	return false;
}

CFeatures* CGUIOctave::set_features(const octave_value_list& vals)
{
	octave_value mat_feat = vals(2);
	CFeatures* f=NULL;
	CIO::message(M_INFO, "start CGUIOctave::set_features\n") ;

	if (!mat_feat.is_empty())
	{
		CIO::message(M_DEBUG, "%d %d %d\n", mat_feat.is_real_matrix(), mat_feat.is_real_scalar(), mat_feat.is_real_nd_array());	
		///octave does not yet support sparse matrices
		//if (mat_feat.is_sparse())
		//{
		//	CIO::message(M_ERROR, "no, no, no. this is not implemented yet\n");
		//}
		//else
		{
			//if (mat_feat.is_real_matrix())
			{
				Matrix m = mat_feat.matrix_value();

				f = new CRealFeatures((LONG) 0);
				INT num_vec = m.cols();
				INT num_feat = m.rows();
				CIO::message(M_DEBUG, "vec: %d feat:%d\n", num_vec, num_feat);
				DREAL* fm = new DREAL[num_vec*num_feat];
				ASSERT(fm);

				for (INT i=0; i<num_vec; i++)
					for (INT j=0; j<num_feat; j++)
						fm[i*num_feat+j]= (double) m(i,j);

				((CRealFeatures*) f)->set_feature_matrix(fm, num_feat, num_vec);
			}
			//else if (mat_feat.is_char_matrix())
			//{

			//	charMatrix cm = mat_feat.char_matrix_value();
			//	f= new CCharFeatures(DNA, (LONG) 0);
			//	INT num_vec = cm.cols();
			//	INT num_feat = cm.rows();
			//	CHAR* fm=new char[num_vec*num_feat+10];
			//	ASSERT(fm);

			//	for (INT i=0; i<num_vec; i++)
			//		for (INT j=0; j<num_feat; j++)
			//			fm[i*num_feat+j]= (char) cm(i,j);

			//	((CCharFeatures*) f)->set_feature_matrix(fm, num_feat, num_vec);
			//}
			/////and so on
			//else
			//	CIO::message(M_ERROR, "not implemented\n");
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
	CIO::message(M_DEBUG, "num: %d\n",num);

	CLabels* label=new CLabels(num);

	CIO::message(M_INFO, "%d\n", label->get_num_labels());

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
	CFeatures* f=gui->guifeatures.get_train_features();
	CFeatures* fe=gui->guifeatures.get_test_features();
	CKernel *k = gui->guikernel.get_kernel();

	if (f && fe)
	{
		int num_vece=fe->get_num_vectors();
		int num_vec=f->get_num_vectors();

		Matrix result = Matrix(num_vec, num_vece);

		for (int i=0; i<num_vec; i++)
			for (int j=0; j<num_vece; j++)
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

	if (kernel_ && (kernel_->get_kernel_type() == K_COMMWORDSTRING))
	{
		CCommWordStringKernel *kernel = (CCommWordStringKernel *) kernel_ ;

		INT len=0 ;
		WORD* dict ;
		DREAL* weights ;
		//FIXMEkernel->get_dictionary(len, dict, weights) ;

		Matrix result = Matrix(len, 2);

		for (int i=0; i<len; i++)
			result(0,i) = dict[i] ;
		for (int i=0; i<len; i++)
			result(1,i) = weights[i] ;

		retvals(0)=result;
		return true;
	}
	return false;
}
#endif
