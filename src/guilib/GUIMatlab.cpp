/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#if defined(HAVE_MATLAB) && !defined(HAVE_SWIG)
#include <stdio.h>
#include <string.h>

#include "guilib/GUIMatlab.h"
#include "gui/TextGUI.h"
#include "gui/GUI.h"

#include "lib/io.h"
#include "base/Version.h"
#include "structure/PlifBase.h"
#include "structure/Plif.h"
#include "structure/PlifArray.h"
#include "structure/DynProg.h"
#include "distributions/hmm/HMM.h"
#include "features/Alphabet.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "kernel/WeightedDegreeStringKernel.h"
#include "kernel/WeightedDegreePositionStringKernel.h"
#include "kernel/CombinedKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "kernel/WeightedCommWordStringKernel.h"
#include "kernel/CustomKernel.h"
#include "kernel/LinearKernel.h"
#include "kernel/SparseLinearKernel.h"
#include "classifier/svm/SVM.h"
#include "lib/Array.h"
#include "lib/Array3.h"

extern CTextGUI* gui;

CGUIMatlab::CGUIMatlab() : CSGObject()
{
}

bool CGUIMatlab::send_command(CHAR* cmd)
{
	return (gui->parse_line(cmd));
}

bool CGUIMatlab::relative_entropy(mxArray* retvals[])
{
	CHMM* pos=gui->guihmm.get_pos();
	CHMM* neg=gui->guihmm.get_neg();

	if (pos && neg)
	{
		if ( (pos->get_M() == neg->get_M()) && (pos->get_N() == neg->get_N()) )
		{
			mxArray* mx_entropy=mxCreateDoubleMatrix(1, pos->get_N(), mxREAL);
			ASSERT(mx_entropy);
			double* _entropy=mxGetPr(mx_entropy);
			ASSERT(_entropy);
			double* p=new double[pos->get_M()];
			double* q=new double[neg->get_M()];

			for (INT i=0; i<pos->get_N(); i++)
			{
				for (INT j=0; j<pos->get_M(); j++)
				{
					p[j]=pos->get_b(i,j);
					q[j]=neg->get_b(i,j);
				}

				_entropy[i]=CMath::relative_entropy(p, q, pos->get_M());
			}
			delete[] p;
			delete[] q;
			retvals[0]=mx_entropy;

			return true;
		}
		else
			SG_ERROR( "pos and neg hmm's differ in number of emissions or states\n");
	}
	else
		SG_ERROR( "set pos and neg hmm first\n");

	return false;
}

bool CGUIMatlab::entropy(mxArray* retvals[])
{
	CHMM* current=gui->guihmm.get_current();

	if (current) 
	{
		mxArray* mx_entropy=mxCreateDoubleMatrix(1, current->get_N(), mxREAL);
		ASSERT(mx_entropy);
		double* _entropy=mxGetPr(mx_entropy);
		double* p=new double[current->get_M()];
		ASSERT(_entropy);

		for (INT i=0; i<current->get_N(); i++)
		{
			for (INT j=0; j<current->get_M(); j++)
			{
				p[j]=current->get_b(i,j);
			}

			_entropy[i]=CMath::entropy(p, current->get_M());
		}

		retvals[0]=mx_entropy;

		return true;
	}
	else
		SG_ERROR( "create hmm first\n");

	return false;
}

bool CGUIMatlab::get_hmm(mxArray* retvals[])
{
	CHMM* h=gui->guihmm.get_current();

	if (h)
	{
		mxArray* mx_p=mxCreateDoubleMatrix(1, h->get_N(), mxREAL);
		mxArray* mx_q=mxCreateDoubleMatrix(1, h->get_N(), mxREAL);
		mxArray* mx_a=mxCreateDoubleMatrix(h->get_N(), h->get_N(), mxREAL);
		mxArray* mx_b=mxCreateDoubleMatrix(h->get_N(), h->get_M(), mxREAL);

		if (mx_p && mx_q && mx_a && mx_b)
		{
			double* p=mxGetPr(mx_p);
			double* q=mxGetPr(mx_q);
			double* a=mxGetPr(mx_a);
			double* b=mxGetPr(mx_b);

			int i,j;
			for (i=0; i< h->get_N(); i++)
			{
				p[i]=h->get_p(i);
				q[i]=h->get_q(i);
			}

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_N(); j++)
					a[i+j*h->get_N()]=h->get_a(i,j);

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_M(); j++)
					b[i+j*h->get_N()]=h->get_b(i,j);

			retvals[0]=mx_p;
			retvals[1]=mx_q;
			retvals[2]=mx_a;
			retvals[3]=mx_b;

			return true;
		}
	}

	return false;
}

bool CGUIMatlab::hmm_likelihood(mxArray* retvals[])
{
	CHMM* h=gui->guihmm.get_current();

	if (h)
	{
		mxArray* mx_p=mxCreateDoubleMatrix(1, 1, mxREAL);
		ASSERT(mx_p);
		double* p=mxGetPr(mx_p);
		ASSERT(p);

		*p=h->model_probability();
		retvals[0]=mx_p;
		return true;
	}
	return false;
}

bool CGUIMatlab::best_path(mxArray* retvals[], int dim)
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
				mxArray* mx_path = mxCreateDoubleMatrix(1, num_feat, mxREAL);
				mxArray* mx_lik = mxCreateDoubleMatrix(1, 1, mxREAL);

				double* lik = mxGetPr(mx_lik);
				double* path = mxGetPr(mx_path);
				T_STATES* s = h->get_path(dim, *lik);

				for (int i=0; i<num_feat; i++)
					path[i]=s[i];

				delete[] s;

				retvals[0]=mx_path;
				retvals[1]=mx_lik;
				return true;
			}
		}
	}
	return false;
}

bool CGUIMatlab::append_hmm(const mxArray* vals[])
{
	CHMM* old_h=gui->guihmm.get_current();
	ASSERT(old_h);

	const mxArray* mx_p=vals[1];
	const mxArray* mx_q=vals[2];
	const mxArray* mx_a=vals[3];
	const mxArray* mx_b=vals[4];

	INT N=mxGetN(mx_a);
	INT M=mxGetN(mx_b);

	if (mx_p && mx_q && mx_a && mx_b)
	{
		CHMM* h=new CHMM(N, M, NULL, gui->guihmm.get_pseudo());
		if (h)
		{
			SG_INFO( "N:%d M:%d p:(%d,%d) q:(%d,%d) a:(%d,%d) b(%d,%d)\n",
					N, M,
					mxGetN(mx_p), mxGetM(mx_p), 
					mxGetN(mx_q), mxGetM(mx_q), 
					mxGetN(mx_a), mxGetM(mx_a), 
					mxGetN(mx_b), mxGetM(mx_b));
			if (
					mxGetN(mx_p) == h->get_N() && mxGetM(mx_p) == 1 &&
					mxGetN(mx_q) == h->get_N() && mxGetM(mx_q) == 1 &&
					mxGetN(mx_a) == h->get_N() && mxGetM(mx_a) == h->get_N() &&
					((INT) mxGetN(mx_b)) == h->get_M() && mxGetM(mx_b) == h->get_N()
			   )
			{
				double* p=mxGetPr(mx_p);
				double* q=mxGetPr(mx_q);
				double* a=mxGetPr(mx_a);
				double* b=mxGetPr(mx_b);

				int i,j;
				for (i=0; i< h->get_N(); i++)
				{
					h->set_p(i, p[i]);
					h->set_q(i, q[i]);
				}

				for (i=0; i< h->get_N(); i++)
					for (j=0; j< h->get_N(); j++)
						h->set_a(i,j, a[i+j*h->get_N()]);

				for (i=0; i< h->get_N(); i++)
					for (j=0; j< h->get_M(); j++)
						h->set_b(i,j, b[i+j*h->get_N()]);

				SG_INFO( "h %d , M: %d\n", h, h->get_M());

				old_h->append_model(h);

				delete h;

				return true;
			}
			else
				SG_ERROR( "model matricies not matching in size\n");
		}
	}
	return false;
}

bool CGUIMatlab::set_hmm(const mxArray* vals[])
{
	const mxArray* mx_p=vals[1];
	const mxArray* mx_q=vals[2];
	const mxArray* mx_a=vals[3];
	const mxArray* mx_b=vals[4];

	INT N=mxGetN(mx_a);
	INT M=mxGetN(mx_b);

	CHMM* h=new CHMM(N, M, NULL, gui->guihmm.get_pseudo());

	if ( mx_p && mx_q && mx_a && mx_b)
	{

		if (h)
		{
			SG_DEBUG( "N:%d M:%d p:(%d,%d) q:(%d,%d) a:(%d,%d) b(%d,%d)\n",
					N, M,
					mxGetM(mx_p), mxGetN(mx_p), 
					mxGetM(mx_q), mxGetN(mx_q), 
					mxGetM(mx_a), mxGetN(mx_a), 
					mxGetM(mx_b), mxGetN(mx_b));

			if (
					mxGetN(mx_p) == h->get_N() && mxGetM(mx_p) == 1 &&
					mxGetN(mx_q) == h->get_N() && mxGetM(mx_q) == 1 &&
					mxGetN(mx_a) == h->get_N() && mxGetM(mx_a) == h->get_N() &&
					((INT) mxGetN(mx_b)) == h->get_M() && mxGetM(mx_b) == h->get_N()
			   )
			{
				double* p=mxGetPr(mx_p);
				double* q=mxGetPr(mx_q);
				double* a=mxGetPr(mx_a);
				double* b=mxGetPr(mx_b);

				int i,j;
				for (i=0; i< h->get_N(); i++)
				{
					h->set_p(i, p[i]);
					h->set_q(i, q[i]);
				}

				for (i=0; i< h->get_N(); i++)
					for (j=0; j< h->get_N(); j++)
						h->set_a(i,j, a[i+j*h->get_N()]);

				for (i=0; i< h->get_N(); i++)
					for (j=0; j< h->get_M(); j++)
						h->set_b(i,j, b[i+j*h->get_N()]);

				gui->guihmm.set_current(h);
				return true;
			}
			else
				SG_ERROR( "model matricies not matching in size\n");
		}
	}

	return false;
}


bool CGUIMatlab::best_path_no_b(const mxArray* vals[], mxArray* retvals[])
{
	const mxArray* mx_p=vals[1];
	const mxArray* mx_q=vals[2];
	const mxArray* mx_a=vals[3];
	const mxArray* mx_max_iter=vals[4];

	INT max_iter = (INT)mxGetScalar(mx_max_iter) ;
	
	INT N=mxGetN(mx_a);

	if ( mx_p && mx_q && mx_a)
	{
		if (
			((INT) mxGetN(mx_p)) == N && mxGetM(mx_p) == 1 &&
			((INT) mxGetN(mx_q)) == N && mxGetM(mx_q) == 1 &&
			((INT) mxGetN(mx_a)) == N && ((INT) mxGetM(mx_a)) == N
			)
		{
			double* p=mxGetPr(mx_p);
			double* q=mxGetPr(mx_q);
			double* a=mxGetPr(mx_a);
			
			CDynProg* h=new CDynProg();
			h->set_N(N) ;
			h->set_p_vector(p, N) ;
			h->set_q_vector(q, N) ;
			h->set_a(a, N, N) ;
			
			INT *my_path = new INT[max_iter] ;
			int best_iter = 0 ;
			DREAL prob = h->best_path_no_b(max_iter, best_iter, my_path) ;

			mxArray* mx_prob = mxCreateDoubleMatrix(1, 1, mxREAL);
			double* p_prob = mxGetPr(mx_prob);
			p_prob[0] = prob ;
			
			mxArray* mx_result = mxCreateDoubleMatrix(1, best_iter+1, mxREAL);
			double* result = mxGetPr(mx_result);
			for (INT i=0; i<best_iter+1; i++)
				result[i]=my_path[i] ;
			
			retvals[0]=mx_prob ;
			retvals[1]=mx_result ;
			delete h ;
			
			return true;
		}
		else
			SG_ERROR( "model matricies not matching in size\n");
	}

	return false;
}

bool CGUIMatlab::best_path_no_b_trans(const mxArray* vals[], mxArray* retvals[])
{
	const mxArray* mx_p=vals[1];
	const mxArray* mx_q=vals[2];
	const mxArray* mx_a_trans=vals[3];
	const mxArray* mx_max_iter=vals[4];
	const mxArray* mx_nbest=vals[5];

	INT max_iter = (INT)mxGetScalar(mx_max_iter) ;
	INT nbest    = (INT)mxGetScalar(mx_nbest) ;
	if (nbest<1)
		return false ;
	if (max_iter<1)
		return false ;
	
	INT N=mxGetN(mx_p);

	if ( mx_p && mx_q && mx_a_trans)
	{
		if (
			((INT) mxGetN(mx_p) == N) && mxGetM(mx_p) == 1 &&
			((INT) mxGetN(mx_q) == N) && mxGetM(mx_q) == 1 &&
			mxGetN(mx_a_trans) == 3
			)
		{
			double* p=mxGetPr(mx_p);
			double* q=mxGetPr(mx_q);
			double* a=mxGetPr(mx_a_trans);
			
			CDynProg* h=new CDynProg() ;
			h->set_N(N) ;
			h->set_p_vector(p, N) ;
			h->set_q_vector(q, N) ;
			h->set_a_trans_matrix(a, mxGetM(mx_a_trans), 3) ;
			
			INT *my_path = new INT[(max_iter+1)*nbest] ;
			memset(my_path, -1, (max_iter+1)*nbest*sizeof(INT)) ;
			
			int max_best_iter = 0 ;
			mxArray* mx_prob = mxCreateDoubleMatrix(1, nbest, mxREAL);
			double* p_prob = mxGetPr(mx_prob);
			
			h->best_path_no_b_trans(max_iter, max_best_iter, nbest, p_prob, my_path) ;

			mxArray* mx_result=mxCreateDoubleMatrix(nbest, max_best_iter+1, mxREAL);
			double* result=mxGetPr(mx_result);
			
			for (INT k=0; k<nbest; k++)
			{
				for (INT i=0; i<max_best_iter+1; i++)
					result[i*nbest+k] = my_path[i+k*(max_iter+1)] ;
			}
			 
			
			retvals[0]=mx_prob ;
			retvals[1]=mx_result ;
			
			delete h ;
			delete[] my_path ;
			return true;
		}
		else
			SG_ERROR( "model matricies not matching in size\n");
	}

	return false;
}


bool CGUIMatlab::best_path_trans(const mxArray* vals[], INT nrhs, mxArray* retvals[])
{
	const mxArray* mx_p=vals[1];
	const mxArray* mx_q=vals[2];
	const mxArray* mx_a_trans=vals[3];
	const mxArray* mx_seq=vals[4];
	const mxArray* mx_pos=vals[5];
	const mxArray* mx_orf_info=vals[6];
	const mxArray* mx_genestr=vals[7];
	const mxArray* mx_penalties=vals[8];
	const mxArray* mx_state_signals=vals[9];
	const mxArray* mx_penalty_info=vals[10];
	const mxArray* mx_nbest=vals[11];
	const mxArray* mx_dict_weights=vals[12];
	const mxArray* mx_use_orf=vals[13];
	const mxArray* mx_mod_words=vals[14] ;
	const mxArray* mx_segment_loss=NULL ;
	const mxArray* mx_segment_ids_mask=NULL ;

	ASSERT(nrhs==17 || nrhs==15) ;

	if (nrhs==17)
	{
		mx_segment_loss=vals[15];
		mx_segment_ids_mask=vals[16];
	} ;

	if ((mxGetM(mx_nbest)!=1) || ((mxGetN(mx_nbest)!=1) && mxGetN(mx_nbest)!=2))
		SG_ERROR( "nbest should be 1x1 or 1x2 \n");
	
	INT nbest    = (INT)mxGetScalar(mx_nbest) ;
	if (nbest<1)
		return false ;
	double *p_n    = mxGetPr(mx_nbest) ;
	INT nother    = 0 ;
	if (mxGetN(mx_nbest)==2)
		nother = (INT) p_n[1] ;
	ASSERT(p_n[0]==nbest) ;
	
	if ( mx_p && mx_q && mx_a_trans && mx_seq && mx_pos && 
		 mx_penalties && mx_state_signals && mx_penalty_info && 
		 mx_orf_info && mx_genestr && mx_dict_weights)
	{
		INT N=mxGetN(mx_p);
		INT M=mxGetN(mx_pos);
		INT P=mxGetN(mx_penalty_info) ;
		INT L=mxGetM(mx_genestr) ;
		INT genestr_num=mxGetN(mx_genestr) ;
		INT D=mxGetM(mx_dict_weights) ;
		INT dict_weigths_num=mxGetN(mx_dict_weights) ;
		

		fprintf(stderr, "genestr_num = %i, L=%i\n", genestr_num, L) ;
		
		if (genestr_num>L)
			SG_ERROR( "more strings than the length of the strings ... it seems likely to be wrongly transposed \n");

		if (!(((INT) mxGetN(mx_p)) == N && mxGetM(mx_p) == 1 &&
			 ((INT) mxGetN(mx_q)) == N && mxGetM(mx_q) == 1 &&
			  ((mxGetN(mx_a_trans) == 3)||(mxGetN(mx_a_trans) == 4)) ))
			SG_ERROR( "model matricies not matching in size \n");

		INT seq_num_dimensions = mxGetNumberOfDimensions(mx_seq) ;
		INT seq_second_dimension = 1 ;
		if (seq_num_dimensions>=2)
			seq_second_dimension = mxGetDimensions(mx_seq)[1] ;
		if (!(((INT) mxGetM(mx_seq)) == N &&
			  seq_second_dimension == ((INT) mxGetN(mx_pos)) && mxGetM(mx_pos)==1 && 
			  ((seq_num_dimensions==2) || (seq_num_dimensions==3))))
			SG_ERROR( "seq and position matrices sizes wrong\n");
		INT seq_third_dimension = 1 ;
		if (seq_num_dimensions==3)
			seq_third_dimension = mxGetDimensions(mx_seq)[2] ;

		INT penalty_num_dimensions = mxGetNumberOfDimensions(mx_penalties) ;
		if (!((penalty_num_dimensions==2) || (penalty_num_dimensions==3)))
			SG_ERROR( "penalties should have 2 or three dimensions (has %i)", penalty_num_dimensions);

		const mwSize* penalty_dimensions = mxGetDimensions(mx_penalties) ;		
		if (!(((INT) penalty_dimensions[0])==N && 
			  ((INT) penalty_dimensions[1])==N))
			SG_ERROR( "size of penalties wrong (%i!=%i or %i!=%i)\n", penalty_dimensions[0], N, penalty_dimensions[1], N);

		INT penalties_dim3 = 1 ;
		if (penalty_num_dimensions==3)
			penalties_dim3 = penalty_dimensions[2] ;
		//fprintf(stderr,"considering up to %i Plifs in a PlifArray\n", penalties_dim3) ;
		ASSERT(penalties_dim3>0) ;

		if (!(((INT) mxGetM(mx_state_signals))==N && 
			  ((INT) mxGetN(mx_state_signals))==seq_third_dimension))
			SG_ERROR( "size of state_signals wrong (%i!=%i or %i!=%i)\n", mxGetM(mx_state_signals), N, mxGetN(mx_state_signals), seq_third_dimension);

		if (!(((mxGetN(mx_dict_weights)==8) || (mxGetN(mx_dict_weights)==16)) && 
			  ((mxIsCell(mx_penalty_info) && mxGetM(mx_penalty_info)==1)
			   || mxIsEmpty(mx_penalty_info))))
			SG_ERROR( "size of dict_weights wrong\n");

		if (!(((mxGetM(mx_mod_words)==6) || (mxGetM(mx_mod_words)==8) || (mxGetM(mx_mod_words)==16)) && mxGetN(mx_mod_words)==2))
			SG_ERROR( "size of mod_words wrong (should be 6x2 or 8x2 or 16x2)\n");

		if (mx_segment_loss!=NULL && (mxGetN(mx_segment_loss)!=2*mxGetM(mx_segment_loss)))
			SG_ERROR( "size of segment_loss wrong\n");

		if (mx_segment_ids_mask!=NULL && ((mxGetM(mx_segment_ids_mask)!=2) ||
										  (((INT) mxGetN(mx_segment_ids_mask))!=M)))
			SG_ERROR( "size of segment_ids_mask wrong\n");

		if (
			mxGetM(mx_use_orf)==1 &&
			mxGetN(mx_use_orf)==1 &&
			((mxIsCell(mx_penalty_info) && mxGetM(mx_penalty_info)==1)
			 || mxIsEmpty(mx_penalty_info))
			)
		{
			double* p=mxGetPr(mx_p);
			double* q=mxGetPr(mx_q);
			double* a=mxGetPr(mx_a_trans);
			INT use_orf=(INT)*mxGetPr(mx_use_orf) ;

			double* seq=mxGetPr(mx_seq) ;

			double* pos_=mxGetPr(mx_pos) ;
			INT * pos = new INT[M] ;
			for (INT i=0; i<M; i++)
				pos[i]=(INT)pos_[i] ;

			double* orf_info_=mxGetPr(mx_orf_info) ;
			INT * orf_info = new INT[2*N] ;
			for (INT i=0; i<2*N; i++)
				orf_info[i]=(INT)orf_info_[i] ;

			CPlif ** PEN = 
				read_penalty_struct_from_cell(mx_penalty_info, P) ;
			if (PEN==NULL && P!=0)
				return false ;
			
			CPlifBase **PEN_matrix = new CPlifBase*[N*N] ;
			double* penalties_array=mxGetPr(mx_penalties) ;
			CArray3<double> penalties(penalties_array, N, N, penalties_dim3, false, false) ;
			
			for (INT i=0; i<N; i++)
				for (INT j=0; j<N; j++)
				{
					CPlifArray * plif_array = new CPlifArray() ;
					CPlif * plif = NULL ;
					plif_array->clear() ;
					for (INT k=0; k<penalties_dim3; k++)
					{
						if (penalties.element(i,j,k)==0)
							continue ;
						INT id = (INT) penalties.element(i,j,k)-1 ;
						if ((id<0 || id>=P) && (id!=-1))
						{
							SG_ERROR( "id out of range\n") ;
							delete_penalty_struct(PEN, P) ;
							return false ;
						}
						plif = PEN[id] ;
						plif_array->add_plif(plif) ;
					}
					if (plif_array->get_num_plifs()==0)
					{
						delete plif_array ;
						PEN_matrix[i+j*N] = NULL ;
					}
					else if (plif_array->get_num_plifs()==1)
					{
						delete plif_array ;
						ASSERT(plif!=NULL) ;
						PEN_matrix[i+j*N] = plif ;
					}
					else
						PEN_matrix[i+j*N] = plif_array ;
				}

			CPlifBase **PEN_state_signal = new CPlifBase*[seq_third_dimension*N] ;
			double* state_signals=mxGetPr(mx_state_signals) ;
			for (INT i=0; i<N*seq_third_dimension; i++)
			{
				INT id = (INT) state_signals[i]-1 ;
				if ((id<0 || id>=P) && (id!=-1))
				{
					SG_ERROR( "id out of range\n") ;
					delete_penalty_struct(PEN, P) ;
					return false ;
				}
				if (id==-1)
					PEN_state_signal[i]=NULL ;
				else
					PEN_state_signal[i]=PEN[id] ;
			} ;

			char * genestr = mxArrayToString(mx_genestr) ;				
			DREAL * dict_weights = mxGetPr(mx_dict_weights) ;
			
			CDynProg* h=new CDynProg();
			h->set_N(N) ;
			h->set_p_vector(p, N) ;
			h->set_q_vector(q, N) ;
			if (mx_segment_ids_mask!=NULL)
				h->set_a_trans_matrix(a, mxGetM(mx_a_trans), mxGetN(mx_a_trans)) ;
			else
				h->set_a_trans_matrix(a, mxGetM(mx_a_trans), 3) ; // segment_id = 0 

			INT* mod_words_array=new INT[mxGetM(mx_mod_words)*mxGetN(mx_mod_words)] ;
			double* mod_words_array_real = mxGetPr(mx_mod_words) ;
			for (INT i=0; i<(INT) (mxGetM(mx_mod_words)*mxGetN(mx_mod_words)); i++)
				mod_words_array[i] = (INT) mod_words_array_real[i] ;
			h->init_mod_words_array(mod_words_array, mxGetM(mx_mod_words), mxGetN(mx_mod_words)) ;

			if (!h->check_svm_arrays())
			{
				SG_ERROR( "svm arrays inconsistent\n") ;
				delete_penalty_struct(PEN, P) ;
				return false ;
			}
			
			INT *my_path = new INT[M*(nbest+nother)] ;
			memset(my_path, -1, M*(nother+nbest)*sizeof(INT)) ;
			INT *my_pos = new INT[M*(nbest+nother)] ;
			memset(my_pos, -1, M*(nbest+nother)*sizeof(INT)) ;
			
			mxArray* mx_prob = mxCreateDoubleMatrix(1, (nbest+nother), mxREAL);
			double* p_prob = mxGetPr(mx_prob);

			if (mx_segment_ids_mask!=NULL)
			{
				h->best_path_set_segment_loss(mxGetPr(mx_segment_loss), mxGetM(mx_segment_loss), mxGetN(mx_segment_loss)) ;
				DREAL *dbuffer = mxGetPr(mx_segment_ids_mask) ;
				INT *ibuffer = new INT[2*M] ;
				for (INT i=0; i<2*M; i++)
					ibuffer[i] = (INT)dbuffer[i] ;
				h->best_path_set_segment_ids_mask(ibuffer, mxGetM(mx_segment_ids_mask), mxGetN(mx_segment_ids_mask)) ;
				delete[] ibuffer ;
			}
			else
			{
				DREAL zero2[2] = {0.0, 0.0} ;
				h->best_path_set_segment_loss(zero2, 2, 1) ;
				//fprintf(stderr, "M=%i\n", M) ;
				INT *zeros = new INT[2*M] ;
				for (INT i=0; i<2*M; i++)
					zeros[i]=0 ;
				h->best_path_set_segment_ids_mask(zeros, 2, M) ;
				delete[] zeros ;
			} ;

			assert(nbest==1 || nbest==2) ;
			assert(nother==0) ;
			if (nbest==1)
				h->best_path_trans<1>(seq, M, pos, orf_info,
								   PEN_matrix, PEN_state_signal, seq_third_dimension, 
								   genestr, L, genestr_num,
								   p_prob, my_path, my_pos, 
								   dict_weights, dict_weigths_num*D, use_orf) ;
			else 
				h->best_path_trans<2>(seq, M, pos, orf_info,
								   PEN_matrix, PEN_state_signal, seq_third_dimension, 
								   genestr, L, genestr_num,
								   p_prob, my_path, my_pos, 
								   dict_weights, dict_weigths_num*D, use_orf) ;

			// clean up 
			delete_penalty_struct(PEN, P) ;
			delete[] PEN_matrix ;
			delete[] pos ;
			delete[] orf_info ;
			delete h ;
			mxFree(genestr) ;

			// transcribe result
			mxArray* mx_my_path=mxCreateDoubleMatrix((nbest+nother), M, mxREAL);
			double* d_my_path=mxGetPr(mx_my_path);
			mxArray* mx_my_pos=mxCreateDoubleMatrix((nbest+nother), M, mxREAL);
			double* d_my_pos=mxGetPr(mx_my_pos);
			
			for (INT k=0; k<(nbest+nother); k++)
				for (INT i=0; i<M; i++)
				{
					d_my_path[i*(nbest+nother)+k] = my_path[i+k*M] ;
					d_my_pos[i*(nbest+nother)+k] = my_pos[i+k*M] ;
				}
			
			retvals[0]=mx_prob ;
			retvals[1]=mx_my_path ;
			retvals[2]=mx_my_pos ;

			delete[] my_path ;
			delete[] my_pos ;

			return true;
		}
		else
			SG_ERROR( "model matricies not matching in size\n");
	}

	return false;
}

bool CGUIMatlab::best_path_trans_deriv(const mxArray* vals[], INT nrhs, mxArray* retvals[], INT nlhs)
{
	const mxArray* mx_my_path=vals[1];
	const mxArray* mx_my_pos=vals[2];
	const mxArray* mx_p=vals[3];
	const mxArray* mx_q=vals[4];
	const mxArray* mx_a_trans=vals[5];
	const mxArray* mx_seq=vals[6];
	const mxArray* mx_pos=vals[7];
	const mxArray* mx_genestr=vals[8];
	const mxArray* mx_penalties=vals[9];
	const mxArray* mx_state_signals=vals[10];
	const mxArray* mx_penalty_info=vals[11];
	const mxArray* mx_dict_weights=vals[12];
	const mxArray* mx_mod_words=vals[13];
	const mxArray* mx_segment_loss=NULL ;
	const mxArray* mx_segment_ids_mask=NULL ;
	
	ASSERT(nrhs==16 || nrhs==14) ;
	ASSERT(nlhs==5 || nlhs==6) ;

	if (nrhs==16)
	{
		mx_segment_loss=vals[14];
		mx_segment_ids_mask=vals[15];
	} ;

	if ( mx_my_path && mx_my_pos && mx_p && mx_q && mx_a_trans && mx_seq && mx_pos && 
		 mx_penalties && mx_state_signals && mx_penalty_info &&
		 mx_genestr && mx_dict_weights)
	{
		INT N=mxGetN(mx_p);
		INT M=mxGetN(mx_pos);
		INT P=mxGetN(mx_penalty_info) ;
		INT genestr_len=mxGetM(mx_genestr) ;
		INT genestr_num=mxGetN(mx_genestr) ;
		INT D=mxGetM(mx_dict_weights) ;
		INT dict_weigths_num = mxGetN(mx_dict_weights) ;
		INT my_seqlen = mxGetN(mx_my_path) ;
				
		if (!(((INT) mxGetN(mx_p)) == N && mxGetM(mx_p) == 1 &&
			  ((INT) mxGetN(mx_q)) == N && mxGetM(mx_q) == 1 &&
			  ((mxGetN(mx_a_trans) == 3) || (mxGetN(mx_a_trans) == 4))))
			SG_ERROR( "model matricies not matching in size \n");

		INT seq_num_dimensions = mxGetNumberOfDimensions(mx_seq) ;
		INT seq_second_dimension = 1 ;
		if (seq_num_dimensions>=2)
			seq_second_dimension = mxGetDimensions(mx_seq)[1] ;
		if (!(((INT) mxGetM(mx_seq)) == N &&
			  seq_second_dimension == ((INT) mxGetN(mx_pos)) && mxGetM(mx_pos)==1 && 
			  ((seq_num_dimensions==2) || (seq_num_dimensions==3))))
			SG_ERROR( "seq and position matrices sizes wrong (%i, %i, %i, %i, %i)\n", 
					  seq_num_dimensions, seq_second_dimension, mxGetN(mx_seq), mxGetM(mx_pos), mxGetN(mx_pos));
		INT seq_third_dimension = 1 ;
		if (seq_num_dimensions==3)
			seq_third_dimension = mxGetDimensions(mx_seq)[2] ;
		
		INT penalty_num_dimensions = mxGetNumberOfDimensions(mx_penalties) ;
		if (!((penalty_num_dimensions==2) || (penalty_num_dimensions==3)))
			SG_ERROR( "penalties should have 2 or 3 dimensions (has %i)", penalty_num_dimensions);

		const mwSize* penalty_dimensions = mxGetDimensions(mx_penalties) ;		
		if (!(((INT) penalty_dimensions[0])==N && 
			  ((INT) penalty_dimensions[1])==N))
			SG_ERROR( "size of penalties wrong (%i!=%i or %i!=%i)\n", penalty_dimensions[0], N, penalty_dimensions[1], N);

		INT penalties_dim3 = 1 ;
		if (penalty_num_dimensions==3)
			penalties_dim3 = penalty_dimensions[2] ;
		//fprintf(stderr,"considering up to %i Plifs in a PlifArray\n", penalties_dim3) ;
		ASSERT(penalties_dim3>0) ;

		if (!(((INT) mxGetM(mx_state_signals))==N && 
			  ((INT) mxGetN(mx_state_signals))==seq_third_dimension))
			SG_ERROR( "size of state_signals wrong (%i!=%i or %i!=%i)\n", mxGetM(mx_state_signals), N, mxGetN(mx_state_signals), seq_third_dimension);
			
		if (!(mxGetN(mx_my_pos)==mxGetN(mx_my_path) &&
			  mxGetM(mx_my_path)==1 &&
			  mxGetM(mx_my_pos)==1))
			SG_ERROR( "size of position and path don't match\n");

		if (!(((mxGetN(mx_dict_weights)==8)||(mxGetN(mx_dict_weights)==16)) && 
			  ((mxIsCell(mx_penalty_info) && mxGetM(mx_penalty_info)==1)
			   || mxIsEmpty(mx_penalty_info))))
			SG_ERROR( "dict_weights or penalty_info wrong\n");

		if (!(((mxGetM(mx_mod_words)==6) || (mxGetM(mx_mod_words)==8) || (mxGetM(mx_mod_words)==16)) && mxGetN(mx_mod_words)==2))
			SG_ERROR( "size mod_words wrong (should be 6x2 or 8x2 or 16x2)\n");

		if (mx_segment_loss!=NULL && (mxGetN(mx_segment_loss)!=2*mxGetM(mx_segment_loss)))
			SG_ERROR( "size of segment_loss wrong\n");

		if (mx_segment_ids_mask!=NULL && ((mxGetM(mx_segment_ids_mask)!=2) ||
										  (((INT) mxGetN(mx_segment_ids_mask))!=M)))
			SG_ERROR( "size of segment_ids_mask wrong\n");

		{
			double* p=mxGetPr(mx_p);
			double* q=mxGetPr(mx_q);
			double* a=mxGetPr(mx_a_trans);
			double* seq=mxGetPr(mx_seq) ;
			double* pos_=mxGetPr(mx_pos) ;

			INT * pos = new INT[M] ;
			for (INT i=0; i<M; i++)
				pos[i]=(INT)pos_[i] ;
			
			CPlif ** PEN = 
				read_penalty_struct_from_cell(mx_penalty_info, P) ;
			if (PEN==NULL && P!=0)
				return false ;

			INT max_plif_id = 0 ;
			INT max_plif_len = 1 ;
			for (INT i=0; i<P; i++)
			{
				ASSERT(PEN[i]->get_id()==i) ;
				if (i>max_plif_id)
					max_plif_id=i ;
				if (PEN[i]->get_plif_len()>max_plif_len)
					max_plif_len=PEN[i]->get_plif_len() ;
			} ;

			CPlifBase **PEN_matrix = new CPlifBase*[N*N] ;
			double* penalties_array=mxGetPr(mx_penalties) ;
			//fprintf(stderr, "N=%i penalties_dim3=%i\n", N, penalties_dim3) ;
			CArray3<double> penalties(penalties_array, N, N, penalties_dim3, false, false) ;

			//INT num_empty=0, num_single=0, num_array=0 ;
		 
			for (INT i=0; i<N; i++)
				for (INT j=0; j<N; j++)
				{
					CPlifArray * plif_array = new CPlifArray() ;
					CPlif * plif = NULL ;
					plif_array->clear() ;
					for (INT k=0; k<penalties_dim3; k++)
					{
						if (penalties.element(i,j,k)==0)
							continue ;
						INT id = (INT) penalties.element(i,j,k)-1 ;
						//fprintf(stderr, "i=%i, j=%i, k=%i, id=%i\n", i, j, k, id) ;

						if ((id<0 || id>=P) && (id!=-1))
						{
							SG_ERROR( "id out of range\n") ;
							delete_penalty_struct(PEN, P) ;
							return false ;
						}
						plif = PEN[id] ;
						plif_array->add_plif(plif) ;
					}
					//fprintf(stderr, "numplifs=%i\n", plif_array->get_num_plifs()) ;
					if (plif_array->get_num_plifs()==0)
					{
						delete plif_array ;
						PEN_matrix[i+j*N] = NULL ;
						//num_empty++ ;
					}
					else if (plif_array->get_num_plifs()==1)
					{
						delete plif_array ;
						ASSERT(plif!=NULL) ;
						PEN_matrix[i+j*N] = plif ;
						//num_single++ ;
					}
					else
					{
						PEN_matrix[i+j*N] = plif_array ;
						//num_array++ ;
					}
				}
			//fprintf(stderr, "num_empty=%i, num_single=%i, num_array=%i\n", num_empty, num_single, num_array) ;

			CPlifBase **PEN_state_signal = new CPlifBase*[seq_third_dimension*N] ;
			double* state_signals=mxGetPr(mx_state_signals) ;
			for (INT i=0; i<seq_third_dimension*N; i++)
			{
				INT id = (INT) state_signals[i]-1 ;
				if ((id<0 || id>=P) && (id!=-1))
				{
					SG_ERROR( "id out of range\n") ;
					delete_penalty_struct(PEN, P) ;
					return false ;
				}
				if (id==-1)
					PEN_state_signal[i]=NULL ;
				else
					PEN_state_signal[i]=PEN[id] ;
			} ;

			char * genestr = mxArrayToString(mx_genestr) ;				
			DREAL * dict_weights = mxGetPr(mx_dict_weights) ;
			
			CDynProg* h=new CDynProg();
			h->set_N(N) ;
			h->set_p_vector(p, N) ;
			h->set_q_vector(q, N) ;
			if (mx_segment_ids_mask!=NULL) 
				h->set_a_trans_matrix(a, mxGetM(mx_a_trans), mxGetN(mx_a_trans)) ;
			else
				h->set_a_trans_matrix(a, mxGetM(mx_a_trans), 3) ; // segment_id = 0 

			INT* mod_words_array=new INT[mxGetM(mx_mod_words)*mxGetN(mx_mod_words)] ;
			double* mod_words_array_real = mxGetPr(mx_mod_words) ;
			for (INT i=0; i<(INT) (mxGetM(mx_mod_words)*mxGetN(mx_mod_words)); i++)
				mod_words_array[i] = (INT) mod_words_array_real[i] ;
			h->init_mod_words_array(mod_words_array, mxGetM(mx_mod_words), mxGetN(mx_mod_words)) ;

			if (!h->check_svm_arrays())
			{
				SG_ERROR( "svm arrays inconsistent\n") ;
				delete_penalty_struct(PEN, P) ;
				return false ;
			}

			INT *my_path = new INT[my_seqlen+1] ;
			memset(my_path, -1, my_seqlen*sizeof(INT)) ;
			INT *my_pos = new INT[my_seqlen+1] ;
			memset(my_pos, -1, my_seqlen*sizeof(INT)) ;

			double* d_my_path=mxGetPr(mx_my_path);
			double* d_my_pos=mxGetPr(mx_my_pos);

			for (INT i=0; i<my_seqlen; i++)
			{
				my_path[i] = (INT)d_my_path[i] ;
				my_pos[i] = (INT)d_my_pos[i] ;
			}

			if (mx_segment_ids_mask!=NULL) 
			{
				h->best_path_set_segment_loss(mxGetPr(mx_segment_loss), mxGetM(mx_segment_loss), mxGetN(mx_segment_loss)) ;
				DREAL *dbuffer = mxGetPr(mx_segment_ids_mask) ;
				INT *ibuffer = new INT[2*M] ;
				for (INT i=0; i<2*M; i++)
					ibuffer[i] = (INT)dbuffer[i] ;
				h->best_path_set_segment_ids_mask(ibuffer, mxGetM(mx_segment_ids_mask), mxGetN(mx_segment_ids_mask)) ;
				delete[] ibuffer ;
			}
			else
			{
				DREAL zero2[2] = {0.0, 0.0} ;
				h->best_path_set_segment_loss(zero2, 2, 1) ;
				//fprintf(stderr, "M=%i\n", M) ;
				INT *zeros = new INT[2*M] ;
				for (INT i=0; i<2*M; i++)
					zeros[i]=0 ;
				h->best_path_set_segment_ids_mask(zeros, 2, M) ;
				delete[] zeros ;
			} ;
						
			mwSize dims_plif[2]={max_plif_id+1, max_plif_len};
			mxArray* Plif_deriv = mxCreateNumericArray(2, dims_plif, mxDOUBLE_CLASS, mxREAL);
			double* p_Plif_deriv = mxGetPr(Plif_deriv);
			CArray2<double> a_Plif_deriv(p_Plif_deriv, max_plif_id+1, max_plif_len, false, false) ;

			mwSize dims_A[2]={N,N};
			mxArray* A_deriv = mxCreateNumericArray(2, dims_A, mxDOUBLE_CLASS, mxREAL);
			double* p_A_deriv = mxGetPr(A_deriv);

			mwSize dims_pq[1]={N};
			mxArray* p_deriv = mxCreateNumericArray(1, dims_pq, mxDOUBLE_CLASS, mxREAL);
			mxArray* q_deriv = mxCreateNumericArray(1, dims_pq, mxDOUBLE_CLASS, mxREAL);
			double* p_p_deriv = mxGetPr(p_deriv);
			double* p_q_deriv = mxGetPr(q_deriv);

			mwSize dims_score[1]={my_seqlen};
			mxArray* my_scores = mxCreateNumericArray(1, dims_score, mxDOUBLE_CLASS, mxREAL);
			double* p_my_scores = mxGetPr(my_scores);

			mxArray* my_losses = mxCreateNumericArray(1, dims_score, mxDOUBLE_CLASS, mxREAL);
			double* p_my_losses = mxGetPr(my_losses);
			
			h->best_path_trans_deriv(my_path, my_pos, p_my_scores, p_my_losses, 
									 my_seqlen, seq, M, pos, 
									 PEN_matrix, PEN_state_signal, seq_third_dimension, 
									 genestr, genestr_len, genestr_num, 
									 dict_weights, dict_weigths_num*D) ;
			
			for (INT i=0; i<N; i++)
			{
				for (INT j=0; j<N; j++)
					p_A_deriv[i+j*N] = h->get_a_deriv(i, j) ;
				p_p_deriv[i]=h->get_p_deriv(i) ;
				p_q_deriv[i]=h->get_q_deriv(i) ;
				}
			
			for (INT id=0; id<=max_plif_id; id++)
			{
				INT len=0 ;
				const DREAL * deriv = PEN[id]->get_cum_derivative(len) ;
				//fprintf(stderr, "len=%i, max_plif_len=%i\n", len, max_plif_len) ;
				ASSERT(len<=max_plif_len) ;
				for (INT j=0; j<max_plif_len; j++)
					a_Plif_deriv.element(id, j)= deriv[j] ;
			}

			// clean up 
			delete_penalty_struct(PEN, P) ;
			delete[] PEN_matrix ;
			delete[] PEN_state_signal ;
			delete[] pos ;
			//delete h ;
			mxFree(genestr) ;

			retvals[0]=p_deriv ;
			retvals[1]=q_deriv ;
			retvals[2]=A_deriv ;
			retvals[3]=Plif_deriv ;
			retvals[4]=my_scores ;
			if (nlhs==6)
				retvals[5]=my_losses ;

			delete[] my_path ;
			delete[] my_pos ;

			return true ;
		}
	}

	return false ;
}

bool CGUIMatlab::best_path_2struct(const mxArray* vals[], mxArray* retvals[])
{
	const mxArray* mx_p=vals[1];
	const mxArray* mx_q=vals[2];
	const mxArray* mx_a_trans=vals[3];
	const mxArray* mx_seq=vals[4];
	const mxArray* mx_pos=vals[5];
	const mxArray* mx_genestr=vals[6];
	const mxArray* mx_penalties=vals[7];
	const mxArray* mx_penalty_info=vals[8];
	const mxArray* mx_nbest=vals[9];
	const mxArray* mx_dict_weights=vals[10];
	const mxArray* mx_segment_sum_weights=vals[11];

	INT nbest    = (INT)mxGetScalar(mx_nbest) ;
	if (nbest<1)
		return false ;
	
	if ( mx_p && mx_q && mx_a_trans && mx_seq && mx_pos && 
		 mx_penalties && mx_penalty_info && 
		 mx_genestr && mx_dict_weights && mx_segment_sum_weights)
	{
		INT N=mxGetN(mx_p);
		INT M=mxGetN(mx_pos);
		INT P=mxGetN(mx_penalty_info) ;
		INT L=mxGetN(mx_genestr) ;
		INT D=mxGetM(mx_dict_weights) ;
		
		//SG_DEBUG( "N=%i, M=%i, P=%i, L=%i, nbest=%i\n", N, M, P, L, nbest) ;
		/*fprintf(stderr,"ok1=%i\n", mxGetN(mx_p) == N && mxGetM(mx_p) == 1 &&
				mxGetN(mx_q) == N && mxGetM(mx_q) == 1 &&
				mxGetN(mx_a_trans) == 3 &&
				mxGetM(mx_seq) == N &&
				mxGetN(mx_seq) == mxGetN(mx_pos) && mxGetM(mx_pos)==1) ;
		fprintf(stderr, "ok2=%i\n", 	mxGetM(mx_penalties)==N && 
				mxGetN(mx_penalties)==N &&
				mxGetM(mx_genestr)==1 &&
				((mxIsCell(mx_penalty_info) && mxGetM(mx_penalty_info)==1)
				 || mxIsEmpty(mx_penalty_info))) ;
		fprintf(stderr, "ok3=%i\n", 				mxGetM(mx_genestr)==1 &&
				((mxIsCell(mx_penalty_info) && mxGetM(mx_penalty_info)==1)
				|| mxIsEmpty(mx_penalty_info))) ;
		fprintf(stderr, "ok4=%i\n", mxGetM(mx_segment_sum_weights)==M &&
		mxGetN(mx_segment_sum_weights)==N ) ;*/
		
		if (
			((INT) mxGetN(mx_p)) == N && mxGetM(mx_p) == 1 &&
			((INT) mxGetN(mx_q)) == N && mxGetM(mx_q) == 1 &&
			mxGetN(mx_a_trans) == 3 &&
			((INT) mxGetM(mx_seq)) == N &&
			mxGetN(mx_seq) == mxGetN(mx_pos) && mxGetM(mx_pos)==1 &&
			((INT) mxGetM(mx_penalties))==N && 
			((INT) mxGetN(mx_penalties))==N &&
			((INT) mxGetM(mx_segment_sum_weights))==N &&
			((INT) mxGetN(mx_segment_sum_weights))==M &&
			mxGetM(mx_genestr)==1 &&
			mxGetN(mx_dict_weights)==1 && 
			((mxIsCell(mx_penalty_info) && mxGetM(mx_penalty_info)==1)
			 || mxIsEmpty(mx_penalty_info))
			)
		{
			double* p=mxGetPr(mx_p);
			double* q=mxGetPr(mx_q);
			double* a=mxGetPr(mx_a_trans);

			double* seq=mxGetPr(mx_seq) ;

			double* pos_=mxGetPr(mx_pos) ;
			INT * pos = new INT[M] ;
			for (INT i=0; i<M; i++)
				pos[i]=(INT)pos_[i] ;

			CPlif ** PEN = 
				read_penalty_struct_from_cell(mx_penalty_info, P) ;
			if (PEN==NULL && P!=0)
				return false ;
			
			CPlifBase **PEN_matrix = new CPlifBase*[N*N] ;
			double* penalties=mxGetPr(mx_penalties) ;
			for (INT i=0; i<N*N; i++)
			{
				INT id = (INT) penalties[i]-1 ;
				if ((id<0 || id>=P) && (id!=-1))
				{
					SG_ERROR( "id out of range\n") ;
					delete_penalty_struct(PEN, P) ;
					return false ;
				}
				if (id==-1)
					PEN_matrix[i]=NULL ;
				else
					PEN_matrix[i]=PEN[id] ;
			} ;
			char * genestr = mxArrayToString(mx_genestr) ;				
			DREAL * dict_weights = mxGetPr(mx_dict_weights) ;
			DREAL * segment_sum_weights = mxGetPr(mx_segment_sum_weights) ;
			
			CDynProg* h=new CDynProg();
			h->set_N(N) ;
			h->set_p_vector(p, N) ;
			h->set_q_vector(q, N) ;
			h->set_a_trans_matrix(a, mxGetM(mx_a_trans), 3) ;
			
			INT *my_path = new INT[(M+1)*nbest] ;
			memset(my_path, -1, (M+1)*nbest*sizeof(INT)) ;
			INT *my_pos = new INT[(M+1)*nbest] ;
			memset(my_pos, -1, (M+1)*nbest*sizeof(INT)) ;
			
			mxArray* mx_prob = mxCreateDoubleMatrix(1, nbest, mxREAL);
			double* p_prob = mxGetPr(mx_prob);
			
			h->best_path_2struct(seq, M, pos, PEN_matrix, genestr, L,
								 nbest, p_prob, my_path, my_pos, dict_weights, 
								 D, segment_sum_weights) ;

			// clean up 
			delete_penalty_struct(PEN, P) ;
			delete[] PEN_matrix ;
			delete[] pos ;
			delete h ;
			mxFree(genestr) ;

			// transcribe result
			mxArray* mx_my_path=mxCreateDoubleMatrix(nbest, M+1, mxREAL);
			double* d_my_path=mxGetPr(mx_my_path);
			mxArray* mx_my_pos=mxCreateDoubleMatrix(nbest, M+1, mxREAL);
			double* d_my_pos=mxGetPr(mx_my_pos);
			
			for (INT k=0; k<nbest; k++)
				for (INT i=0; i<M+1; i++)
				{
					d_my_path[i*nbest+k] = my_path[i+k*(M+1)] ;
					d_my_pos[i*nbest+k] = my_pos[i+k*(M+1)] ;
				}
			
			retvals[0]=mx_prob ;
			retvals[1]=mx_my_path ;
			retvals[2]=mx_my_pos ;

			delete[] my_path ;
			delete[] my_pos ;

			return true;
		}
		else
			SG_ERROR( "model matricies not matching in size\n");
	}

	return false;
}


bool CGUIMatlab::best_path_trans_simple(const mxArray* vals[], mxArray* retvals[])
{
	const mxArray* mx_p=vals[1];
	const mxArray* mx_q=vals[2];
	const mxArray* mx_a_trans=vals[3];
	const mxArray* mx_seq=vals[4];
	const mxArray* mx_nbest=vals[5];

	INT nbest    = (INT)mxGetScalar(mx_nbest) ;
	if (nbest<1)
		return false ;
	
	if ( mx_p && mx_q && mx_a_trans && mx_seq)
	{
		INT N=mxGetN(mx_p);
		INT M=mxGetN(mx_seq);
		
		if (
			((INT) mxGetN(mx_p)) == N && mxGetM(mx_p) == 1 &&
			((INT) mxGetN(mx_q)) == N && mxGetM(mx_q) == 1 &&
			mxGetN(mx_a_trans) == 3 &&
			((INT) mxGetM(mx_seq)) == N)
		{
			double* p=mxGetPr(mx_p);
			double* q=mxGetPr(mx_q);
			double* a=mxGetPr(mx_a_trans);

			double* seq=mxGetPr(mx_seq) ;
			
			CDynProg* h=new CDynProg();
			h->set_N(N) ;
			h->set_p_vector(p, N) ;
			h->set_q_vector(q, N) ;
			h->set_a_trans_matrix(a, mxGetM(mx_a_trans), 3) ;
			
			INT *my_path = new INT[M*nbest] ;
			memset(my_path, -1, M*nbest*sizeof(INT)) ;
			
			mxArray* mx_prob = mxCreateDoubleMatrix(1, nbest, mxREAL);
			double* p_prob = mxGetPr(mx_prob);
			
			h->best_path_trans_simple(seq, M, nbest, p_prob, my_path) ;

			// clean up 
			delete h ;

			// transcribe result
			mxArray* mx_my_path=mxCreateDoubleMatrix(nbest, M, mxREAL);
			double* d_my_path=mxGetPr(mx_my_path);
			
			for (INT k=0; k<nbest; k++)
				for (INT i=0; i<M; i++)
				{
					d_my_path[i*nbest+k] = my_path[i+k*M] ;
				}
			
			retvals[0]=mx_prob ;
			retvals[1]=mx_my_path ;

			delete[] my_path ;

			return true;
		}
		else
			SG_ERROR( "model matricies not matching in size\n");
	}

	return false;
}


/*bool CGUIMatlab::model_prob_no_b_trans(const mxArray* vals[], mxArray* retvals[])
{
	const mxArray* mx_p=vals[1];
	const mxArray* mx_q=vals[2];
	const mxArray* mx_a_trans=vals[3];
	const mxArray* mx_max_iter=vals[4];

	INT max_iter = (INT)mxGetScalar(mx_max_iter) ;
	if (max_iter<1)
		return false ;
	
	INT N=mxGetN(mx_p);

	if ( mx_p && mx_q && mx_a_trans)
	{
		if (
			mxGetN(mx_p) == N && mxGetM(mx_p) == 1 &&
			mxGetN(mx_q) == N && mxGetM(mx_q) == 1 &&
			mxGetN(mx_a_trans) == 3
			)
		{
			double* p=mxGetPr(mx_p);
			double* q=mxGetPr(mx_q);
			double* a=mxGetPr(mx_a_trans);
			
			CHMM* h=new CHMM(N, p, q, mxGetM(mx_a_trans), a);
			
			mxArray* mx_prob = mxCreateDoubleMatrix(1, max_iter, mxREAL);
			double* p_prob = mxGetPr(mx_prob);
			
			h->model_prob_no_b_trans(max_iter, p_prob) ;

			retvals[0]=mx_prob ;
			
			delete h ;
			return true;
		}
		else
			SG_ERROR( "model matricies not matching in size\n");
	}

	return false;
}
*/


bool CGUIMatlab::hmm_classify(mxArray* retvals[])
{
	CFeatures* f=gui->guifeatures.get_test_features();
	if (f)
	{
		int num_vec=f->get_num_vectors();

		mxArray* mx_result=mxCreateDoubleMatrix(1, num_vec, mxREAL);
		double* result=mxGetPr(mx_result);
		CLabels* l=gui->guihmm.classify();

		for (int i=0; i<num_vec; i++)
			result[i]=l->get_label(i);

		delete l;

		retvals[0]=mx_result;
		return true;
	}
	return false;
}

bool CGUIMatlab::hmm_classify_example(mxArray* retvals[], int idx)
{
	mxArray* mx_result=mxCreateDoubleMatrix(1, 1, mxREAL);
	double* result=mxGetPr(mx_result);
	*result=gui->guihmm.classify_example(idx);
	retvals[0]=mx_result;
	return true;
}

bool CGUIMatlab::one_class_hmm_classify(mxArray* retvals[], bool linear)
{
	CFeatures* f=gui->guifeatures.get_test_features();
	if (f)
	{
		int num_vec=f->get_num_vectors();

		mxArray* mx_result = mxCreateDoubleMatrix(1, num_vec, mxREAL);
		double* result     = mxGetPr(mx_result);
		
		CLabels* l         = NULL ;
		if (!linear)
		  l=gui->guihmm.one_class_classify();
		else
		  l=gui->guihmm.linear_one_class_classify();

		for (int i=0; i<num_vec; i++)
			result[i]=l->get_label(i);

		delete l;

		retvals[0]=mx_result;
		return true;
	}
	return false;
}

bool CGUIMatlab::one_class_hmm_classify_example(mxArray* retvals[], int idx)
{
	mxArray* mx_result=mxCreateDoubleMatrix(1, 1, mxREAL);
	double* result=mxGetPr(mx_result);
	*result=gui->guihmm.one_class_classify_example(idx);
	retvals[0]=mx_result;
	return true;
}

bool CGUIMatlab::get_classifier(mxArray* retvals[])
{
	DREAL* bias=NULL;
	DREAL* weights=NULL;
	INT rows=0;
	INT cols=0;
	INT brows=0;
	INT bcols=0;


	if (gui->guiclassifier.get_trained_classifier(weights, rows, cols, bias, brows, bcols))
	{
		mxArray* mx_w=mxCreateDoubleMatrix(rows, cols, mxREAL);
		mxArray* mx_b=mxCreateDoubleMatrix(brows, bcols, mxREAL);

		if (mx_w && mx_b)
		{
			double* b=mxGetPr(mx_b);
			double* w=mxGetPr(mx_w);

			memcpy(b, bias, bcols*brows*sizeof(DREAL));
			memcpy(w, weights, cols*rows*sizeof(DREAL));
			delete[] weights;

			retvals[0]=mx_b;
			retvals[1]=mx_w;
			return true;
		}
	}

	return false;
}

bool CGUIMatlab::set_svm(const mxArray* vals[])
{
	CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();

	if (svm)
	{
		const mxArray* mx_b=vals[1];
		const mxArray* mx_alphas=vals[2];

		if (
				mx_b && mx_alphas &&
				mxGetN(mx_b) == 1 && mxGetM(mx_b) == 1 &&
				mxGetN(mx_alphas) == 2
			)
		{
			double* b=mxGetPr(mx_b);
			double* alphas=mxGetPr(mx_alphas);

			svm->create_new_model(mxGetM(mx_alphas));
			svm->set_bias(*b);

			for (int i=0; i< svm->get_num_support_vectors(); i++)
			{
				svm->set_alpha(i, alphas[i]);
				svm->set_support_vector(i, (int) alphas[i+svm->get_num_support_vectors()]);
			}

			return true;
		}
	}
	else
		SG_ERROR( "no svm object available\n") ;

	return false;
}

bool CGUIMatlab::classify(mxArray* retvals[])
{
	CFeatures* f=gui->guifeatures.get_test_features();
	if (f)
	{
		int num_vec=f->get_num_vectors();

		CLabels* l=gui->guiclassifier.classify();

		if (!l)
		{
			SG_ERROR( "classify failed\n") ;
			return false ;
		} ;

		mxArray* mx_result=mxCreateDoubleMatrix(1, num_vec, mxREAL);
		double* result=mxGetPr(mx_result);
		for (int i=0; i<num_vec; i++)
			result[i]=l->get_label(i);
		delete l;

		retvals[0]=mx_result;
		return true;
	}
	return false;
}


bool CGUIMatlab::classify_example(mxArray* retvals[], int idx)
{
	mxArray* mx_result=mxCreateDoubleMatrix(1, 1, mxREAL);
	retvals[0]=mx_result;
	double* result=mxGetPr(mx_result);
	
	if (!gui->guiclassifier.classify_example(idx, result[0]))
	  {
	    SG_ERROR( "classify_example failed\n") ;
	    return false ;
	  } ;
	
	return true;
}

bool CGUIMatlab::set_plugin_estimate(const mxArray* vals[])
{
  int num_params = mxGetM(vals[1]) ;
  ASSERT(mxGetN(vals[1])==2) ;
  double* result=mxGetPr(vals[1]);
  DREAL* pos_params = result;
  DREAL* neg_params = &(result[num_params]) ;
  double* p_size=mxGetPr(vals[2]);
  int seq_length = (int)p_size[0] ;
  int num_symbols = (int)p_size[1] ;
  ASSERT(num_params == seq_length*num_symbols) ;

  gui->guipluginestimate.get_estimator()->set_model_params(pos_params, neg_params, seq_length, num_symbols) ;

  return true;
}

bool CGUIMatlab::get_plugin_estimate(mxArray* retvals[])
{
  DREAL* pos_params, * neg_params ;
  int num_params = 0, seq_length=0, num_symbols=0 ;

  if (!gui->guipluginestimate.get_estimator()->get_model_params(pos_params, neg_params, seq_length, num_symbols))
    return false ;

  num_params = seq_length * num_symbols ;

  mxArray* mx_result=mxCreateDoubleMatrix(num_params, 2, mxREAL);
  double* result=mxGetPr(mx_result);
  for (int i=0; i<num_params; i++)
    result[i] = pos_params[i] ;
  for (int i=0; i<num_params; i++)
    result[i+num_params] = neg_params[i] ;
  
  retvals[0]=mx_result;

  mxArray* mx_size=mxCreateDoubleMatrix(1, 2, mxREAL);
  double* p_size=mxGetPr(mx_size);
  p_size[0]=(double)seq_length ;
  p_size[1]=(double)num_symbols ;
  
  retvals[1]=mx_size ;
  return true;
}

bool CGUIMatlab::plugin_estimate_classify(mxArray* retvals[])
{
	CFeatures* f=gui->guifeatures.get_test_features();
	if (f)
	{
		int num_vec=f->get_num_vectors();

		mxArray* mx_result=mxCreateDoubleMatrix(1, num_vec, mxREAL);
		double* result=mxGetPr(mx_result);
		CLabels* l=gui->guipluginestimate.classify();

		for (int i=0; i<num_vec; i++)
			result[i]=l->get_label(i);

		delete l;

		retvals[0]=mx_result;
		return true;
	}
	return false;
}

bool CGUIMatlab::plugin_estimate_classify_example(mxArray* retvals[], int idx)
{
	mxArray* mx_result=mxCreateDoubleMatrix(1, 1, mxREAL);
	double* result=mxGetPr(mx_result);
	*result=gui->guipluginestimate.classify_example(idx);
	retvals[0]=mx_result;
	return true;
}

bool CGUIMatlab::get_features(mxArray* retvals[], CFeatures* f)
{
	if (f)
	{
		mxArray* mx_feat=NULL;

		switch (f->get_feature_class())
		{
			case C_SIMPLE:
				switch (f->get_feature_type())
				{
					case F_DREAL:
						{
							INT num_feat=((CRealFeatures*) f)->get_num_features();
							INT num_vec=((CRealFeatures*) f)->get_num_vectors();
							mx_feat=mxCreateDoubleMatrix(num_feat, num_vec, mxREAL);

							if (mx_feat)
							{
								double* feat=mxGetPr(mx_feat);

								for (INT i=0; i<num_vec; i++)
								{
									INT num_vfeat;
									bool free_vec;
									DREAL* vec=((CRealFeatures*) f)->get_feature_vector(i, num_vfeat, free_vec);
									ASSERT(num_vfeat==num_feat);

									for (INT j=0; j<num_vfeat; j++)
										feat[num_feat*i+j]= (double) vec[j];
									((CRealFeatures*) f)->free_feature_vector(vec, i, free_vec);
								}
							}
						}
						break;
					case F_WORD:
						{
							INT num_feat=((CWordFeatures*) f)->get_num_features();
							INT num_vec=((CWordFeatures*) f)->get_num_vectors();
							mx_feat=mxCreateNumericMatrix(num_feat, num_vec, mxUINT16_CLASS, mxREAL);
							if (mx_feat)
							{
								WORD* feat=(WORD*) mxGetData(mx_feat);

								for (INT i=0; i<num_vec; i++)
								{
									INT num_vfeat;
									bool free_vec;
									WORD* vec=((CWordFeatures*) f)->get_feature_vector(i, num_vfeat, free_vec);
									ASSERT(num_feat==num_vfeat);
									for (INT j=0; j<num_vfeat; j++)
										feat[((CWordFeatures*) f)->get_num_vectors()*j+i]= vec[j];
									((CWordFeatures*) f)->free_feature_vector(vec, i, free_vec);
								}
							}
						}
						break;
					case F_SHORT:
						{
							INT num_feat=((CShortFeatures*) f)->get_num_features();
							INT num_vec=((CShortFeatures*) f)->get_num_vectors();
							mx_feat=mxCreateNumericMatrix(num_feat, num_vec, mxINT16_CLASS, mxREAL);

							if (mx_feat)
							{
								SHORT* feat=(SHORT*) mxGetData(mx_feat);

								for (INT i=0; i<num_vec; i++)
								{
									INT num_vfeat;
									bool free_vec;
									SHORT* vec=((CShortFeatures*) f)->get_feature_vector(i, num_vfeat, free_vec);
									ASSERT(num_feat==num_vfeat);
									for (INT j=0; j<num_vfeat; j++)
										feat[num_vfeat*i+j]= vec[j];
									((CShortFeatures*) f)->free_feature_vector(vec, i, free_vec);
								}
							}
						}
						break;
					case F_CHAR:
						{
							INT num_feat=((CCharFeatures*) f)->get_num_features();
							INT num_vec=((CCharFeatures*) f)->get_num_vectors();
							mx_feat=mxCreateNumericMatrix(num_feat, num_vec, mxCHAR_CLASS, mxREAL);

							if (mx_feat)
							{
								CHAR* feat=(CHAR*) mxGetData(mx_feat);

								for (INT i=0; i<num_vec; i++)
								{
									INT num_vfeat;
									bool free_vec;
									CHAR* vec=((CCharFeatures*) f)->get_feature_vector(i, num_vfeat, free_vec);
									ASSERT(num_feat==num_vfeat);
									for (INT j=0; j<num_vfeat; j++)
										feat[num_vfeat*i+j]= vec[j];
									((CCharFeatures*) f)->free_feature_vector(vec, i, free_vec);
								}
							}
						}
						break;
					case F_BYTE:
						{
							INT num_feat=((CByteFeatures*) f)->get_num_features();
							INT num_vec=((CByteFeatures*) f)->get_num_vectors();
							mx_feat=mxCreateNumericMatrix(num_feat, num_vec, mxUINT8_CLASS, mxREAL);

							if (mx_feat)
							{
								BYTE* feat=(BYTE*) mxGetData(mx_feat);

								for (INT i=0; i<num_vec; i++)
								{
									INT num_vfeat;
									bool free_vec;
									BYTE* vec=((CByteFeatures*) f)->get_feature_vector(i, num_vfeat, free_vec);
									ASSERT(num_feat==num_vfeat);
									for (INT j=0; j<num_vfeat; j++)
										feat[num_vfeat*i+j]= vec[j];
									((CByteFeatures*) f)->free_feature_vector(vec, i, free_vec);
								}
							}
						}
						break;
					default:
						SG_ERROR( "not implemented\n");
				}
				break;
			case C_SPARSE:
				switch (f->get_feature_type())
				{
					case F_DREAL:
						{
							long nnz=((CSparseFeatures<DREAL>*) f)->get_num_nonzero_entries();
							int num_vec=f->get_num_vectors();
							int num_feat=((CSparseFeatures<DREAL>*) f)->get_num_features();

							SG_DEBUG("sparse matrix has %d rows, %d cols and %d nnz elemements\n", num_feat, num_vec, nnz);
							mx_feat=mxCreateSparse(num_feat,num_vec, nnz, mxREAL);
							double* A  = mxGetPr(mx_feat);
							mwIndex* iA = mxGetIr(mx_feat);
							mwIndex* kA = mxGetJc(mx_feat);

							INT offs=0;
							for (INT i=0; i<num_vec; i++)
							{
								INT len=0;
								bool dofree=false;
								TSparseEntry<DREAL>* fv=((CSparseFeatures<DREAL>*) f)->get_sparse_feature_vector(i, len, dofree);
								kA[i]=offs;
								for (INT j=0; j<len; j++)
								{
									A[offs]=fv[j].entry;
									iA[offs]=fv[j].feat_index;
									offs++;
								}
								((CSparseFeatures<DREAL>*) f)->free_feature_vector(fv, len, dofree);
							}
							ASSERT(offs==nnz);
							kA[num_vec]=nnz;
						}
						break;
					default:
						SG_ERROR("not implemented\n");
				};
				break;
			case C_STRING:
				switch (f->get_feature_type())
				{
					case F_CHAR:
						{
							int num_vec=f->get_num_vectors();
							mx_feat=mxCreateCellMatrix(1,num_vec);

							if (mx_feat)
							{
								for (int i=0; i<num_vec; i++)
								{
									INT len=0;
									CHAR* fv=((CStringFeatures<CHAR>*) f)->get_feature_vector(i, len);

									if (len>0)
									{
										char* str=new char[len+1];
										strncpy(str, fv, len);
										str[len]='\0';

										mxSetCell(mx_feat,i,mxCreateString(str));
									}
									else
										mxSetCell(mx_feat,i,mxCreateString(""));
								}
							}
						}
						break;
					case F_WORD:
						{
							INT num_vec=((CRealFeatures*) f)->get_num_vectors();
							mx_feat=mxCreateCellMatrix(1,num_vec);

							if (mx_feat)
							{

								for (INT i=0; i<num_vec; i++)
								{
									INT len=0;
									WORD* fv=((CStringFeatures<WORD>*) f)->get_feature_vector(i, len);

									if (len>0)
									{
										mxArray* mx_element=mxCreateNumericMatrix(1, len, mxUINT16_CLASS, mxREAL);
										ASSERT(mx_element);
										WORD* element=(WORD*) mxGetData(mx_element);
										ASSERT(element);
										memcpy(element, fv, len*sizeof(WORD));
										mxSetCell(mx_feat,i,mx_element);
									}
								}
							}
						}
						break;

					default:
						SG_ERROR("not implemented\n");
				};
				break;
			default:
				SG_ERROR( "not implemented\n");
		}
		if (mx_feat)
			retvals[0]=mx_feat;

		return (mx_feat!=NULL);
	}

	return false;
}

bool CGUIMatlab::set_custom_kernel(const mxArray* vals[], bool source_is_diag, bool dest_is_diag)
{
	const mxArray* mx_kernel=vals[1];

	if (mxIsDouble(mx_kernel))
	{
		const double* km=mxGetPr(mx_kernel);

		CCustomKernel* k=(CCustomKernel*)gui->guikernel.get_kernel();
		if  (k && k->get_kernel_type() == K_COMBINED)
		{
			SG_DEBUG( "identified combined kernel\n") ;
			k = (CCustomKernel*)((CCombinedKernel*)k)->get_last_kernel() ;
		}

		if (k && k->get_kernel_type() == K_CUSTOM)
		{
			if (source_is_diag && dest_is_diag && (mxGetN(mx_kernel) == mxGetM(mx_kernel)) )
				return k->set_diag_kernel_matrix_from_diag(km, mxGetN(mx_kernel));
			else if (!source_is_diag && dest_is_diag && (mxGetN(mx_kernel) == mxGetM(mx_kernel)) )
				return k->set_diag_kernel_matrix_from_full(km, mxGetN(mx_kernel));
			else if (!source_is_diag && !dest_is_diag)
				return k->set_full_kernel_matrix_from_full(km, mxGetM(mx_kernel), mxGetN(mx_kernel));
			else
				SG_ERROR("not defined / general error\n");
		}  else
			SG_ERROR( "not a custom kernel\n") ;
	}
	else
		SG_ERROR("kernel matrix must by given as double matrix\n");

	return false;
}

CFeatures* CGUIMatlab::set_features(const mxArray* vals[], int nrhs)
{
	const mxArray* mx_feat=vals[2];
	CFeatures* f=NULL;
	SG_INFO( "start CGUIMatlab::set_features\n") ;

	if (mx_feat)
	{
		if (mxIsSparse(mx_feat) && mxIsNumeric(mx_feat))
		{
			f= new CSparseFeatures<DREAL>(0);
			INT num_feat = mxGetM(mx_feat);
			INT num_vec = mxGetN(mx_feat);

			long nnz=mxGetNzmax(mx_feat);
			double* A = mxGetPr(mx_feat);
			mwIndex* iA = mxGetIr(mx_feat);
			mwIndex* kA = mxGetJc(mx_feat);

			SG_DEBUG("sparse matrix has %d rows, %d cols and %d nnz elemements\n", num_feat, num_vec, nnz);
			TSparse<DREAL>* sfm= new TSparse<DREAL>[num_vec];
			ASSERT(sfm);

			long offs=0;
			for (INT i=0; i<num_vec; i++)
			{
				INT len=kA[i+1]-kA[i];
				sfm[i].vec_index=i;
				sfm[i].num_feat_entries=len;
				
				if (len>0)
				{
					sfm[i].features= new TSparseEntry<DREAL>[len];
					ASSERT(sfm[i].features);
				}
				else
					sfm[i].features=0;

				for (INT j=0; j<len; j++)
				{
					sfm[i].features[j].entry=A[offs];
					sfm[i].features[j].feat_index=iA[offs];
					offs++;
				}
			}
			ASSERT(offs==nnz);
			((CSparseFeatures<DREAL>*) f)->set_sparse_feature_matrix(sfm, num_feat, num_vec);
		}
		else
		{
			if (mxIsDouble(mx_feat))
			{
				f= new CRealFeatures(0);
				INT num_vec=mxGetN(mx_feat);
				INT num_feat=mxGetM(mx_feat);
				DREAL* fm=new DREAL[num_vec*num_feat];
				ASSERT(fm);
				double* feat=mxGetPr(mx_feat);

				SG_DEBUG("dense matrix has %d rows, %d cols\n", num_feat, num_vec);
				for (INT i=0; i<num_vec; i++)
				  for (INT j=0; j<num_feat; j++)
				    fm[i*num_feat+j]=feat[i*num_feat+j];
				
				((CRealFeatures*) f)->set_feature_matrix(fm, num_feat, num_vec);
			}
			else if (mxIsChar(mx_feat))
			{
				if (nrhs==4)
				{
					INT len=0;
					CHAR* al = CGUIMatlab::get_mxString(vals[3], len);
					CAlphabet* alpha = new CAlphabet(al, len);
					ASSERT(alpha);

					INT num_vec=mxGetN(mx_feat);
					INT num_feat=mxGetM(mx_feat);
					T_STRING<CHAR>* sc=new T_STRING<CHAR>[num_vec];
					ASSERT(sc);
					mxChar* c=mxGetChars(mx_feat);
					ASSERT(c);

					int maxlen=num_feat;

					for (int i=0; i<num_vec; i++)
					{
						sc[i].length=num_feat;
						sc[i].string=new CHAR[num_feat];
						ASSERT(sc[i].string)

						for (INT j=0; j<num_feat; j++)
							sc[i].string[j]=(CHAR) c[((LONG) num_feat)*i+j];
					}
					f= new CStringFeatures<CHAR>(alpha);
					ASSERT(f);

					if (!((CStringFeatures<CHAR>*) f)->set_features(sc, num_vec, maxlen))
					{
						delete f;
						f=NULL;
					}
				}
				else
					SG_ERROR( "please specify alphabet!\n");
			}
			else if (mxIsClass(mx_feat,"uint8") || mxIsClass(mx_feat, "int8"))
			{
				if (nrhs==4)
				{
					INT len=0;
					CHAR* al = CGUIMatlab::get_mxString(vals[3], len);
					CAlphabet* alpha = new CAlphabet(al, len);
					ASSERT(alpha);

					INT num_vec=mxGetN(mx_feat);
					INT num_feat=mxGetM(mx_feat);
					T_STRING<BYTE>* sc=new T_STRING<BYTE>[num_vec];
					ASSERT(sc);
					BYTE* c= (BYTE*) mxGetData(mx_feat);

					int maxlen=num_feat;

					for (int i=0; i<num_vec; i++)
					{
						sc[i].length=num_feat;
						sc[i].string=new BYTE[num_feat];
						ASSERT(sc[i].string)

						for (INT j=0; j<num_feat; j++)
							sc[i].string[j]=(BYTE) c[((LONG) num_feat)*i+j];
					}

					f= new CStringFeatures<BYTE>(alpha);
					ASSERT(f);

					if (!((CStringFeatures<BYTE>*) f)->set_features(sc, num_vec, maxlen))
					{
						delete f;
						f=NULL;
					}
				}
				else
					SG_ERROR( "please specify alphabet!\n");
			}			
			else if (mxIsCell(mx_feat))
			{
				int num_vec=mxGetNumberOfElements(mx_feat);

				ASSERT(num_vec>=1 && mxGetCell(mx_feat, 0));


				if (mxIsChar(mxGetCell(mx_feat, 0)))
				{
					if (nrhs==4)
					{
						INT len=0;
						CHAR* al = CGUIMatlab::get_mxString(vals[3], len);
						CAlphabet* alpha = new CAlphabet(al, len);
						T_STRING<CHAR>* sc=new T_STRING<CHAR>[num_vec];
						ASSERT(alpha);
						ASSERT(sc);

						int maxlen=0;

						for (int i=0; i<num_vec; i++)
						{
							mxArray* e=mxGetCell(mx_feat, i);
							ASSERT(e && mxIsChar(e));
							sc[i].string=get_mxString(e, len);

							if (sc[i].string)
							{
								sc[i].length=len;
								maxlen=CMath::max(maxlen, sc[i].length);
							}
							else
							{
								SG_WARNING( "string with index %d has zero length\n", i+1);
								sc[i].length=0;
							}
						}

						f= new CStringFeatures<CHAR>(alpha);
						ASSERT(f);

						if (!((CStringFeatures<CHAR>*) f)->set_features(sc, num_vec, maxlen))
						{
							delete f;
							f=NULL;
						}
					}
					else
						SG_ERROR( "please specify alphabet!\n");
				}
				else if (mxIsClass(mxGetCell(mx_feat, 0), "uint8") || mxIsClass(mxGetCell(mx_feat, 0), "int8"))
				{
					if (nrhs==4)
					{
						INT len=0;
						CHAR* al = CGUIMatlab::get_mxString(vals[3], len);
						CAlphabet* alpha = new CAlphabet(al, len);
						T_STRING<BYTE>* sc=new T_STRING<BYTE>[num_vec];
						ASSERT(alpha);

						int maxlen=0;

						for (int i=0; i<num_vec; i++)
						{
							mxArray* e=mxGetCell(mx_feat, i);
							ASSERT(e && (mxIsClass(e, "uint8") || mxIsClass(e, "int8")));
							INT _len=0;
							sc[i].string=get_mxBytes(e, _len);
							if (sc[i].string)
							{
								sc[i].length=_len;
								maxlen=CMath::max(maxlen, sc[i].length);
							}
							else
							{
								SG_WARNING( "string with index %d has zero length\n", i+1);
								sc[i].length=0;
							}
						}

						f= new CStringFeatures<BYTE>(alpha);
						ASSERT(f);

						if (!((CStringFeatures<BYTE>*) f)->set_features(sc, num_vec, maxlen))
						{
							delete f;
							f=NULL;
						}
					}
					else
						SG_ERROR( "please specify alphabet!\n");
				}

			}
			else
				SG_ERROR( "not implemented\n");
		}
	}
	return f;
}

bool CGUIMatlab::from_position_list(const mxArray* vals[], int nrhs)
{
	INT skip=0;
	INT slen=0;
	CHAR* target=CGUIMatlab::get_mxString(vals[1], slen);
	const mxArray* mx_winsz=vals[2];
	const mxArray* mx_shift=vals[3];
	const mxArray* mx_skip=NULL;

	if (nrhs==5)
		mx_skip=vals[4];

	ASSERT(mx_winsz && mxIsDouble(mx_winsz) && 
			mxGetN(mx_winsz) == 1 && mxGetM(mx_winsz) == 1);
	ASSERT(mx_shift && mxIsDouble(mx_shift) && mxGetM(mx_shift) == 1);
	SG_DEBUG("shifts: N:%d M:%d\n",mxGetN(mx_shift),mxGetM(mx_shift));

	INT winsize= (INT) (*mxGetPr(mx_winsz));
	INT num_shift=mxGetN(mx_shift);
	if (mx_skip && mxIsDouble(mx_skip) && mxGetN(mx_skip) == 1 && mxGetM(mx_skip) == 1)
		skip= (INT) (*mxGetPr(mx_skip));
	SG_DEBUG("winsize: %d num_shifts:%d skip:%d\n", winsize, num_shift, skip);

	double* shifts= mxGetPr(mx_shift);
	ASSERT(shifts);
	CDynamicArray<INT> positions(mxGetN(mx_shift)+1);

	for (INT i=0; i<num_shift; i++)
	{
		INT s= (INT) floor(shifts[i]);
		if (floor(shifts[i]) - shifts[i] != 0.0)
		{
			SG_ERROR("error in shifts array[%d]=%f - input should be integer\n", i, shifts[i]);
			return false;
		}
		positions.set_element(s, i);
	}

	if ( target && (!strncmp(target, "TRAIN", 5) || 
				!strncmp(target, "TEST", 4) ))
	{
		CFeatures* features=NULL;

		if (!strncmp(target, "TRAIN", 5))
		{
			gui->guifeatures.invalidate_train();
			features= gui->guifeatures.get_train_features();
		}
		else if (!strncmp(target, "TEST", 4))
		{
			gui->guifeatures.invalidate_test();
			features= gui->guifeatures.get_test_features();
		}
		delete[] target;

		ASSERT(features);
		if (((CFeatures*) features)->get_feature_class() == C_COMBINED)
			features= ((CCombinedFeatures*) features)->get_last_feature_obj();

		ASSERT(features);
		ASSERT(((CFeatures*) features)->get_feature_class() == C_STRING);

		switch (features->get_feature_type())
		{
			case F_CHAR:
				return ( ((CStringFeatures<CHAR>*) features)->obtain_by_position_list(winsize,
							&positions, skip) > 0);
			case F_BYTE:
				return ( ((CStringFeatures<BYTE>*) features)->obtain_by_position_list(winsize,
							&positions, skip) > 0);
			case F_WORD:
				return ( ((CStringFeatures<WORD>*) features)->obtain_by_position_list(winsize,
							&positions, skip) > 0);
			case F_ULONG:
				return ( ((CStringFeatures<ULONG>*) features)->obtain_by_position_list(winsize,
							&positions, skip) > 0);
			default:
				SG_SERROR("unsupported string features type\n");
				return false;
		}
	}
	else
		SG_SERROR("usage is sg('from_position_list', 'TRAIN|TEST', [list], skip)");

	return false;
}

bool CGUIMatlab::get_version(mxArray* retvals[])
{
	mxArray* mx_ver=mxCreateDoubleMatrix(1, 1, mxREAL);

	if (mx_ver)
	{
		double* ver=mxGetPr(mx_ver);

		*ver=0;
		*ver = version.get_version_revision();

		retvals[0]=mx_ver;
		return true;
	}

	return false;
}

bool CGUIMatlab::get_svm_objective(mxArray* retvals[])
{
	mxArray* mx_v=mxCreateDoubleMatrix(1, 1, mxREAL);
	CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();

	if (mx_v && svm)
	{
		double* v=mxGetPr(mx_v);

		*v = svm->get_objective();

		retvals[0]=mx_v;
		return true;
	}
	else
		SG_ERROR( "no svm set\n");

	return false;
}

bool CGUIMatlab::get_labels(mxArray* retvals[], CLabels* label)
{
	if (label)
	{
		mxArray* mx_lab=mxCreateDoubleMatrix(1, label->get_num_labels(), mxREAL);

		if (mx_lab)
		{
			double* lab=mxGetPr(mx_lab);

			for (int i=0; i< label->get_num_labels(); i++)
				lab[i]=label->get_label(i);

			retvals[0]=mx_lab;
			return true;
		}
	}

	return false;
}

CLabels* CGUIMatlab::set_labels(const mxArray* vals[])
{
	const mxArray* mx_lab=vals[2];

	if (mx_lab && mxGetM(mx_lab)==1 )
	{
		CLabels* label=new CLabels(mxGetN(mx_lab));

		double* lab=mxGetPr(mx_lab);

		SG_INFO( "%d\n", label->get_num_labels());

		for (int i=0; i<label->get_num_labels(); i++)
			if (!label->set_label(i, lab[i]))
				SG_ERROR( "weirdo ! %d %d\n", label->get_num_labels(), i);

		return label;
	}

	return NULL;
}


CHAR* CGUIMatlab::get_mxString(const mxArray* s, INT& len, bool zero_terminate)
{
	if ( (mxIsChar(s)) && (mxGetM(s)==1) )
	{
		len = mxGetN(s);
		CHAR* string=NULL;
		if (zero_terminate)
			string=new CHAR[len+1];
		else
			string=new CHAR[len];
		ASSERT(string);
		mxChar* c=mxGetChars(s);
		ASSERT(c);
		for (INT i=0; i<len; i++)
			string[i]= (CHAR) (c[i]);

		if (zero_terminate)
			string[len]='\0';

		return string;
	}
	else
		return NULL;
}

BYTE* CGUIMatlab::get_mxBytes(const mxArray* s, INT& len)
{
	if ( (mxIsClass(s, "uint8") || mxIsClass(s, "int8")) && (mxGetM(s)==1) )
	{
		len = mxGetN(s);
		BYTE* bytes=new BYTE[len];
		ASSERT(bytes);
		BYTE* c=(BYTE *) mxGetData(s);
		ASSERT(c);
		for (INT i=0; i<len; i++)
			bytes[i]= (BYTE) (c[i]);

		return bytes;
	}
	else
		return NULL;
}

bool CGUIMatlab::get_kernel_matrix(mxArray* retvals[])
{
	CKernel* k = gui->guikernel.get_kernel();
	if (k && k->get_rhs() && k->get_lhs())
	{
		int num_vec1=k->get_lhs()->get_num_vectors();
		int num_vec2=k->get_rhs()->get_num_vectors();

		mxArray* mx_result=mxCreateDoubleMatrix(num_vec1, num_vec2, mxREAL);
		double* result=mxGetPr(mx_result);

		k->get_kernel_matrix_real(num_vec1, num_vec2, result);
		retvals[0]=mx_result;
		return true;
	}
	else
		SG_ERROR( "no kernel defined");

	return false;
}

bool CGUIMatlab::get_distance_matrix(mxArray* retvals[])
{
	CDistance* k = gui->guidistance.get_distance();
	if (k && k->get_rhs() && k->get_lhs())
	{
		int num_vec1=k->get_lhs()->get_num_vectors();
		int num_vec2=k->get_rhs()->get_num_vectors();

		mxArray* mx_result=mxCreateDoubleMatrix(num_vec1, num_vec2, mxREAL);
		double* result=mxGetPr(mx_result);

		k->get_distance_matrix_real(num_vec1, num_vec2, result);
		retvals[0]=mx_result;
		return true;
	}
	else
		SG_ERROR( "no kernel defined");

	return false;
}

bool CGUIMatlab::get_kernel_optimization(mxArray* retvals[], const mxArray* vals[], int nrhs)
{
	CKernel *kernel = gui->guikernel.get_kernel() ;
	
	if (kernel)
	{
		switch (kernel->get_kernel_type())
		{
			case K_WEIGHTEDDEGREEPOS:
				{
					INT max_order=0;
					if ((nrhs==2) && (mxIsDouble(vals[1])))
						max_order= (INT) *mxGetPr(vals[1]);
					else
						SG_ERROR("parameter missing\n");

					if ((max_order < 1) || (max_order > 12))
					{
						SG_WARNING( "max_order out of range 1..12 (%d). setting to 1\n", max_order);
						max_order=1;
					}

					CWeightedDegreePositionStringKernel* k = (CWeightedDegreePositionStringKernel *) kernel;

					CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
					ASSERT(svm);
					INT num_suppvec = svm->get_num_support_vectors();
					INT* sv_idx    = new INT[num_suppvec];
					DREAL* sv_weight = new DREAL[num_suppvec];
					INT num_feat=-1;
					INT num_sym=-1;

					for (INT i=0; i<num_suppvec; i++)
					{
						sv_idx[i]    = svm->get_support_vector(i);
						sv_weight[i] = svm->get_alpha(i);
					}

					DREAL* position_weights;
					position_weights = k->extract_w( max_order, num_feat, num_sym, NULL,
							num_suppvec, sv_idx, sv_weight);
					mxArray* mx_result ;
					mx_result=mxCreateDoubleMatrix(num_sym, num_feat, mxREAL);
					double* result=mxGetPr(mx_result);

					for (int i=0; i<num_feat; i++)
					{
						for (int j=0; j<num_sym; j++)
							result[i*num_sym+j] = position_weights[i*num_sym+j] ;
					}
					retvals[0]=mx_result;
					return true;
				}
			case K_COMMWORDSTRING:
			case K_WEIGHTEDCOMMWORDSTRING:
				{
					CCommWordStringKernel *k = (CCommWordStringKernel *) kernel ;

					INT len=0 ;
					DREAL* weights ;
					k->get_dictionary(len, weights) ;

					mxArray* mx_result=mxCreateDoubleMatrix(len, 1, mxREAL);
					double* result=mxGetPr(mx_result);
					for (int i=0; i<len; i++)
						result[i]=weights[i] ;

					retvals[0]=mx_result;
					return true;
				}
			case K_LINEAR:
				{
					CLinearKernel *k = (CLinearKernel *) kernel ;

					INT len=0 ;
					const double* weights = k->get_normal(len);

					mxArray* mx_result=mxCreateDoubleMatrix(len, 1, mxREAL);
					double* result=mxGetPr(mx_result);
					for (int i=0; i<len; i++)
						result[i]=weights[i] ;

					retvals[0]=mx_result;
					return true;
				}
			case K_SPARSELINEAR:
				{
					CSparseLinearKernel *k = (CSparseLinearKernel *) kernel ;

					INT len=0 ;
					const double* weights = k->get_normal(len);

					mxArray* mx_result=mxCreateDoubleMatrix(len, 1, mxREAL);
					double* result=mxGetPr(mx_result);
					for (int i=0; i<len; i++)
						result[i]=weights[i] ;

					retvals[0]=mx_result;
					return true;
				}
			default:
				break;
		}
	}
	return false;
}

bool CGUIMatlab::compute_by_subkernels(mxArray* retvals[])
{
	CKernel *kernel_ = gui->guikernel.get_kernel() ;
	
	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
	{
		CWeightedDegreeStringKernel *kernel = (CWeightedDegreeStringKernel *) kernel_;

		if (!kernel->is_tree_initialized())
		{
			SG_ERROR( "optimization not initialized\n") ;
			return false ;
		}
		if (!kernel->get_rhs())
		{
			SG_ERROR( "no rhs\n") ;
			return false ;
		}
		INT num    = kernel->get_rhs()->get_num_vectors() ;
		INT degree = -1;
		INT len = -1;
		// get degree & len
		kernel->get_degree_weights(degree, len);

		if (len==0)
			len=1;
		
		mxArray* mx_result=mxCreateDoubleMatrix(degree*len, num, mxREAL);
		double* result=mxGetPr(mx_result);

		for (int i=0; i<num*degree*len; i++)
			result[i]=0 ;
		
		for (int i=0; i<num; i++)
			kernel->compute_by_tree(i,&result[i*degree*len]) ;
		
		retvals[0]=mx_result;
		return true;
	}
	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
	{
		CWeightedDegreePositionStringKernel *kernel = (CWeightedDegreePositionStringKernel *) kernel_ ;
		
		if (!kernel->is_tree_initialized())
		{
			SG_ERROR( "optimization not initialized\n") ;
			return false ;
		}
		if (!kernel->get_rhs())
		{
			SG_ERROR( "no rhs\n") ;
			return false ;
		}
		INT num    = kernel->get_rhs()->get_num_vectors() ;
		INT degree = -1;
		INT len = -1;
		// get degree & len
		kernel->get_degree_weights(degree, len);

		if (len==0)
			len=1;
		
		mxArray* mx_result=mxCreateDoubleMatrix(degree*len, num, mxREAL);
		double* result=mxGetPr(mx_result);

		for (int i=0; i<num*degree*len; i++)
			result[i]=0 ;
		
		for (int i=0; i<num; i++)
			kernel->compute_by_tree(i,&result[i*degree*len]) ;
		
		retvals[0]=mx_result;
		return true;
	}
	return false;
}


bool CGUIMatlab::get_subkernel_weights(mxArray* retvals[])
{
	CKernel *kernel_ = gui->guikernel.get_kernel() ;
	INT degree=-1;
	INT length=-1;
	
	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
	{
		CWeightedDegreeStringKernel *kernel = (CWeightedDegreeStringKernel *) kernel_;

		const DREAL* weights = kernel->get_degree_weights(degree, length) ;
		if (length == 0)
			length = 1;
		
		mxArray* mx_result=mxCreateDoubleMatrix(degree, length, mxREAL);
		double* result=mxGetPr(mx_result);
		
		for (int i=0; i<degree*length; i++)
			result[i] = weights[i] ;
		
		retvals[0]=mx_result;
		return true;
	}

	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
	{
		CWeightedDegreePositionStringKernel *kernel = (CWeightedDegreePositionStringKernel *) kernel_;

		const DREAL* weights = kernel->get_degree_weights(degree, length) ;
		if (length == 0)
			length = 1;
		
		mxArray* mx_result=mxCreateDoubleMatrix(degree, length, mxREAL);
		double* result=mxGetPr(mx_result);
		
		for (int i=0; i<degree*length; i++)
			result[i] = weights[i] ;
		
		retvals[0]=mx_result;
		return true;
	}
	if (kernel_ && (kernel_->get_kernel_type() == K_COMBINED))
	{
		CCombinedKernel *kernel = (CCombinedKernel *) kernel_ ;
		INT num_weights = -1 ;
		const DREAL* weights = kernel->get_subkernel_weights(num_weights) ;
		
		mxArray* mx_result=mxCreateDoubleMatrix(1, num_weights, mxREAL);
		double* result=mxGetPr(mx_result);
		
		for (int i=0; i<num_weights; i++)
			result[i] = weights[i] ;
		
		retvals[0]=mx_result;
		return true;
	}
	return false;
}

bool CGUIMatlab::get_last_subkernel_weights(mxArray* retvals[])
{
	CKernel *ckernel = gui->guikernel.get_kernel() ;
	if (ckernel && (ckernel->get_kernel_type() == K_COMBINED))
	{
		CKernel *kernel_ = ((CCombinedKernel*)ckernel)->get_last_kernel() ;
		
		INT degree=-1;
		INT length=-1;
		
		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
		{
			CWeightedDegreeStringKernel *kernel = (CWeightedDegreeStringKernel *) kernel_;

			const DREAL* weights = kernel->get_degree_weights(degree, length) ;
			if (length == 0)
				length = 1;
			
			mxArray* mx_result=mxCreateDoubleMatrix(degree, length, mxREAL);
			double* result=mxGetPr(mx_result);
			
			for (int i=0; i<degree*length; i++)
				result[i] = weights[i] ;
			
			retvals[0]=mx_result;
			return true;
		}
		
		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
		{
			CWeightedDegreePositionStringKernel *kernel = (CWeightedDegreePositionStringKernel *) kernel_;
			const DREAL* weights = kernel->get_degree_weights(degree, length) ;
			if (length == 0)
				length = 1;
			
			mxArray* mx_result=mxCreateDoubleMatrix(degree, length, mxREAL);
			double* result=mxGetPr(mx_result);
			
			for (int i=0; i<degree*length; i++)
				result[i] = weights[i] ;
			
			retvals[0]=mx_result;
			return true;
		}
		if (kernel_ && (kernel_->get_kernel_type() == K_COMBINED))
		{
			CCombinedKernel *kernel = (CCombinedKernel *) kernel_ ;
			INT num_weights = -1 ;
			const DREAL* weights = kernel->get_subkernel_weights(num_weights) ;
			
			mxArray* mx_result=mxCreateDoubleMatrix(1, num_weights, mxREAL);
			double* result=mxGetPr(mx_result);
			
			for (int i=0; i<num_weights; i++)
				result[i] = weights[i] ;
			
			retvals[0]=mx_result;
			return true;
		}
	}
	
	SG_ERROR( "get_last_subkernel_weights only works for combined kernels") ;
	return false;
}

bool CGUIMatlab::get_WD_position_weights(mxArray* retvals[])
{
	CKernel *kernel_ = gui->guikernel.get_kernel() ;
	INT length=-1;
	
	if (kernel_ && (kernel_->get_kernel_type() == K_COMBINED))
		kernel_=((CCombinedKernel*)kernel_)->get_last_kernel() ;

	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
	{
		CWeightedDegreeStringKernel *kernel
			= (CWeightedDegreeStringKernel *) kernel_;

		const DREAL* position_weights = kernel->get_position_weights(length) ;
		mxArray* mx_result ;
		if (position_weights==NULL)
			mx_result=mxCreateDoubleMatrix(1, 0, mxREAL);
		else
		{
			mx_result=mxCreateDoubleMatrix(1, length, mxREAL);
			double* result=mxGetPr(mx_result);
			
			for (int i=0; i<length; i++)
				result[i] = position_weights[i] ;
		}
		retvals[0]=mx_result;
		return true;
	}
	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
	{
		CWeightedDegreePositionStringKernel *kernel
			= (CWeightedDegreePositionStringKernel *) kernel_;

		const DREAL* position_weights = kernel->get_position_weights(length) ;
		mxArray* mx_result ;
		if (position_weights==NULL)
			mx_result=mxCreateDoubleMatrix(1, 0, mxREAL);
		else
		{
			mx_result=mxCreateDoubleMatrix(1, length, mxREAL);
			double* result=mxGetPr(mx_result);
			
			for (int i=0; i<length; i++)
				result[i] = position_weights[i] ;
		}
		retvals[0]=mx_result;
		return true;
	}
	return false;
}

bool CGUIMatlab::get_SPEC_scoring(mxArray* retvals[], INT max_order)
{
	CKernel* k= gui->guikernel.get_kernel() ;

	if (k && ((k->get_kernel_type() == K_COMMWORDSTRING) ||
				(k->get_kernel_type() == K_WEIGHTEDCOMMWORDSTRING)) )
	{
		CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
		ASSERT(svm);
		INT num_suppvec = svm->get_num_support_vectors();
		INT* sv_idx    = new INT[num_suppvec];
		DREAL* sv_weight = new DREAL[num_suppvec];
		INT num_feat=-1;
		INT num_sym=-1;

		for (INT i=0; i<num_suppvec; i++)
		{
			sv_idx[i]    = svm->get_support_vector(i);
			sv_weight[i] = svm->get_alpha(i);
		}

		if ((max_order < 1) || (max_order > 8))
		{
			SG_WARNING( "max_order out of range 1..8 (%d). setting to 1\n", max_order);
			max_order=1;
		}
		DREAL* position_weights=NULL;

		switch (k->get_kernel_type())
		{
			case K_COMMWORDSTRING:
				position_weights = ((CCommWordStringKernel*) k)->compute_scoring(
						max_order, num_feat, num_sym, NULL,
						num_suppvec, sv_idx, sv_weight);
				break;
			case K_WEIGHTEDCOMMWORDSTRING:
				position_weights = ((CWeightedCommWordStringKernel*) k)->compute_scoring(
						max_order, num_feat, num_sym, NULL,
						num_suppvec, sv_idx, sv_weight);
				break;
			default:
				SG_ERROR("unsupported kernel\n");
		}
		
		mxArray* mx_result ;
		mx_result=mxCreateDoubleMatrix(num_sym, num_feat, mxREAL);
		double* result=mxGetPr(mx_result);

		for (int i=0; i<num_feat; i++)
		{
			for (int j=0; j<num_sym; j++)
				result[i*num_sym+j] = position_weights[i*num_sym+j] ;
		}
		retvals[0]=mx_result;
		return true;
	}
	else
		SG_ERROR( "one cannot compute a scoring using this kernel function\n");
	return false;
}

bool CGUIMatlab::compute_poim_wd(mxArray* retvals[], const mxArray* vals[], int nrhs)
{
	INT max_order=0;
	DREAL* distribution = NULL;

	if (mxIsDouble(vals[1]))
		max_order=(INT) *mxGetPr(vals[1]);
	else
		return false;

	if (mxIsDouble(vals[2]))
		distribution = mxGetPr(vals[2]);
	else
		return false;

	CKernel *k= gui->guikernel.get_kernel() ;

	if (k && (k->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
	{
		INT num_feat=-1;
		INT num_sym=-1;

		CStringFeatures<CHAR>* sf= (CStringFeatures<CHAR>*) (((CWeightedDegreePositionStringKernel*) k)->get_lhs());
		ASSERT(sf);
		num_feat=sf->get_max_vector_length();
		num_sym= (INT) sf->get_num_symbols();

		if (((INT) mxGetN(vals[2]))!=num_feat || ((INT) mxGetM(vals[2])) != num_sym)
			SG_ERROR("distribution should have (seqlen x num_sym) elements"
					"(seqlen: %d vs. %d symbols: %d vs. %d)\n", num_feat,
					mxGetN(vals[2]), num_sym, mxGetM(vals[2]));

		CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
		ASSERT(svm);
		INT num_suppvec = svm->get_num_support_vectors();
		INT* sv_idx    = new INT[num_suppvec];
		ASSERT(sv_idx);
		DREAL* sv_weight = new DREAL[num_suppvec];
		ASSERT(sv_weight);

		for (INT i=0; i<num_suppvec; i++)
		{
			sv_idx[i]    = svm->get_support_vector(i);
			sv_weight[i] = svm->get_alpha(i);
		}

		if ((max_order < 1) || (max_order > 12))
		{
		  //SG_WARNING( "max_order out of range 1..12 (%d).\n", max_order);
			//SG_WARNING( "max_order out of range 1..12 (%d). setting to 1.\n", max_order);
			//max_order=1;
		}
		DREAL* position_weights;
		position_weights = ((CWeightedDegreePositionStringKernel*) k)->compute_POIM(
				max_order, num_feat, num_sym, NULL,
				num_suppvec, sv_idx, sv_weight, distribution);
		mxArray* mx_result ;
		mx_result=mxCreateDoubleMatrix(num_sym, num_feat, mxREAL);
		double* result=mxGetPr(mx_result);

		for (int i=0; i<num_feat; i++)
		{
			for (int j=0; j<num_sym; j++)
				result[i*num_sym+j] = position_weights[i*num_sym+j] ;
		}
		retvals[0]=mx_result;

		delete[] sv_idx;
		delete[] sv_weight;
		return true;
	}
	else
		SG_ERROR( "one cannot compute POIM using this kernel function\n");

	return false;
}

bool CGUIMatlab::get_WD_scoring(mxArray* retvals[], INT max_order)
{
	CKernel *k= gui->guikernel.get_kernel() ;

	if (k && (k->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
	{
		CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
		ASSERT(svm);
		INT num_suppvec = svm->get_num_support_vectors();
		INT* sv_idx    = new INT[num_suppvec];
		DREAL* sv_weight = new DREAL[num_suppvec];
		INT num_feat=-1;
		INT num_sym=-1;

		for (INT i=0; i<num_suppvec; i++)
		{
			sv_idx[i]    = svm->get_support_vector(i);
			sv_weight[i] = svm->get_alpha(i);
		}

		if ((max_order < 1) || (max_order > 12))
		{
			SG_WARNING( "max_order out of range 1..12 (%d). setting to 1\n", max_order);
			max_order=1;
		}
		DREAL* position_weights;
		position_weights = ((CWeightedDegreePositionStringKernel*) k)->compute_scoring(
				max_order, num_feat, num_sym, NULL,
				num_suppvec, sv_idx, sv_weight);
		mxArray* mx_result ;
		mx_result=mxCreateDoubleMatrix(num_sym, num_feat, mxREAL);
		double* result=mxGetPr(mx_result);

		for (int i=0; i<num_feat; i++)
		{
			for (int j=0; j<num_sym; j++)
				result[i*num_sym+j] = position_weights[i*num_sym+j] ;
		}
		retvals[0]=mx_result;
		return true;
	}
	else
		SG_ERROR( "one cannot compute a scoring using this kernel function\n");
	return false;
}

bool CGUIMatlab::get_WD_consensus(mxArray* retvals[])
{
	CKernel *k= gui->guikernel.get_kernel() ;

	if (k && k->get_kernel_type() == K_WEIGHTEDDEGREEPOS)
	{
		CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
		ASSERT(svm);
		INT num_suppvec = svm->get_num_support_vectors();
		INT* sv_idx    = new INT[num_suppvec];
		DREAL* sv_weight = new DREAL[num_suppvec];
		INT num_feat=-1;

		for (INT i=0; i<num_suppvec; i++)
		{
			sv_idx[i]    = svm->get_support_vector(i);
			sv_weight[i] = svm->get_alpha(i);
		}

		CHAR* consensus = ((CWeightedDegreePositionStringKernel*) k)->compute_consensus(
				num_feat, num_suppvec, sv_idx, sv_weight);
		mxArray* mx_result ;
		mwSize mx_num_feat=(mwSize) num_feat;
		mx_result=mxCreateCharArray(1, &mx_num_feat);
		mxChar* result= (mxChar*) mxGetPr(mx_result);

		for (INT i=0; i<num_feat; i++)
			result[i]=(mxChar) consensus[i];

		delete[] consensus;

		retvals[0]=mx_result;
		return true;
	}
	else
		SG_ERROR( "one cannot compute a scoring using this kernel function\n");
	return false;
}

bool CGUIMatlab::get_SPEC_consensus(mxArray* retvals[])
{
	CKernel *k= gui->guikernel.get_kernel() ;

	if (k && k->get_kernel_type() == K_COMMWORDSTRING)
	{
		CSVM* svm=(CSVM*) gui->guiclassifier.get_classifier();
		ASSERT(svm);
		INT num_suppvec = svm->get_num_support_vectors();
		INT* sv_idx    = new INT[num_suppvec];
		DREAL* sv_weight = new DREAL[num_suppvec];
		INT num_feat=-1;

		for (INT i=0; i<num_suppvec; i++)
		{
			sv_idx[i]    = svm->get_support_vector(i);
			sv_weight[i] = svm->get_alpha(i);
		}

		CHAR* consensus = ((CCommWordStringKernel*) k)->compute_consensus(
				num_feat, num_suppvec, sv_idx, sv_weight);
		mxArray* mx_result ;
		mwSize mx_num_feat=(mwSize) num_feat;
		mx_result=mxCreateCharArray(1, &mx_num_feat);
		mxChar* result= (mxChar*) mxGetPr(mx_result);

		for (INT i=0; i<num_feat; i++)
			result[i]=(mxChar) consensus[i];

		delete[] consensus;

		retvals[0]=mx_result;
		return true;
	}
	else
		SG_ERROR( "one cannot compute a scoring using this kernel function\n");
	return false;
}

bool CGUIMatlab::set_subkernel_weights(const mxArray* mx_arg)
{
	CKernel *kernel_ = gui->guikernel.get_kernel() ;
	
	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
	{
		CWeightedDegreeStringKernel *kernel
			= (CWeightedDegreeStringKernel *) kernel_;
		INT degree = kernel->get_degree() ;
		if (((INT) mxGetM(mx_arg))!=degree || mxGetN(mx_arg)<1)
		{
			SG_ERROR( "dimension mismatch (should be de(seq_length | 1) x degree)\n") ;
			return false ;
		}

		INT len = mxGetN(mx_arg);

		if (len ==  1)
			len = 0;

		return kernel->set_weights(mxGetPr(mx_arg), mxGetM(mx_arg), len);
		
	}

	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
	{
		CWeightedDegreePositionStringKernel *kernel
			= (CWeightedDegreePositionStringKernel *) kernel_;
		INT degree = kernel->get_degree() ;
		if (((INT) mxGetM(mx_arg))!=degree || mxGetN(mx_arg)<1)
		{
			SG_ERROR( "dimension mismatch (should be (seq_length | 1) x degree)\n") ;
			return false ;
		}
		
		INT len = mxGetN(mx_arg);
		
		if (len ==  1)
			len = 0;
		
		return kernel->set_weights(mxGetPr(mx_arg), mxGetM(mx_arg), len);
	}

	// all other kernels
	CKernel *kernel = kernel_ ;
	INT num_subkernels = kernel->get_num_subkernels() ;
	if (mxGetM(mx_arg)!=1 || ((INT) mxGetN(mx_arg))!=num_subkernels)
	{
		SG_ERROR( "dimension mismatch (should be 1 x num_subkernels)\n") ;
		return false ;
	}
		
	kernel->set_subkernel_weights(mxGetPr(mx_arg), mxGetN(mx_arg));
	return true ;
}

bool CGUIMatlab::set_subkernel_weights_combined(const mxArray** mx_arg)
{
	CKernel *ckernel = gui->guikernel.get_kernel() ;

	if ((mxGetM(mx_arg[2])!=1) || (mxGetN(mx_arg[2])!=1))
		return false ;
	
	INT kernel_idx = (INT) mxGetScalar(mx_arg[2]) ;
	SG_DEBUG( "using kernel_idx=%i\n", kernel_idx) ;
	
	if (ckernel && (ckernel->get_kernel_type() == K_COMBINED))
	{
		CKernel *kernel_ = ((CCombinedKernel*)ckernel)->get_kernel(kernel_idx) ;
		ASSERT(kernel_!=NULL) ;
		
		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
		{
			CWeightedDegreeStringKernel *kernel
				= (CWeightedDegreeStringKernel *) kernel_;
			INT degree = kernel->get_degree() ;
			if (((INT) mxGetM(mx_arg[1]))!=degree || mxGetN(mx_arg[1])<1)
			{
				SG_ERROR( "dimension mismatch (should be de(seq_length | 1) x degree)\n") ;
				return false ;
			}
			
			INT len = mxGetN(mx_arg[1]);
			
			if (len ==  1)
				len = 0;
			
			return kernel->set_weights(mxGetPr(mx_arg[1]), mxGetM(mx_arg[1]), len);
			
		}
		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
		{
			CWeightedDegreePositionStringKernel *kernel = (CWeightedDegreePositionStringKernel *) kernel_;
			INT degree = kernel->get_degree() ;
			if (((INT) mxGetM(mx_arg[1]))!=degree || mxGetN(mx_arg[1])<1)
			{
				SG_ERROR( "dimension mismatch (should be (seq_length | 1) x degree)\n") ;
				return false ;
			}
			INT len = mxGetN(mx_arg[1]);
			
			if (len ==  1)
				len = 0;
			
			return kernel->set_weights(mxGetPr(mx_arg[1]), mxGetM(mx_arg[1]), len);
		}
		// all other kernels
		CKernel *kernel = kernel_ ;
		INT num_subkernels = kernel->get_num_subkernels() ;
		if (mxGetM(mx_arg[1])!=1 || ((INT) mxGetN(mx_arg[1]))!=num_subkernels)
		{
			SG_ERROR( "dimension mismatch (should be 1 x num_subkernels)\n") ;
			return false ;
		}
		
		kernel->set_subkernel_weights(mxGetPr(mx_arg[1]), mxGetN(mx_arg[1]));
		return true ;
	}

	SG_ERROR( "set_last_subkernel_weights only works for combined kernels") ;
	return false ;
}

bool CGUIMatlab::set_last_subkernel_weights(const mxArray* mx_arg)
{
	CKernel *ckernel = gui->guikernel.get_kernel() ;
	if (ckernel && (ckernel->get_kernel_type() == K_COMBINED))
	{
		CKernel *kernel_ = ((CCombinedKernel*)ckernel)->get_last_kernel() ;
		
		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
		{
			CWeightedDegreeStringKernel *kernel
				= (CWeightedDegreeStringKernel *) kernel_;
			INT degree = kernel->get_degree() ;
			if (((INT) mxGetM(mx_arg))!=degree || mxGetN(mx_arg)<1)
			{
				SG_ERROR( "dimension mismatch (should be de(seq_length | 1) x degree)\n") ;
				return false ;
			}
			
			INT len = mxGetN(mx_arg);
			
			if (len ==  1)
				len = 0;
			
			return kernel->set_weights(mxGetPr(mx_arg), mxGetM(mx_arg), len);
			
		}
		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
		{
			CWeightedDegreePositionStringKernel *kernel
				= (CWeightedDegreePositionStringKernel *) kernel_;
			INT degree = kernel->get_degree() ;
			if (((INT) mxGetM(mx_arg))!=degree || mxGetN(mx_arg)<1)
			{
				SG_ERROR( "dimension mismatch (should be (seq_length | 1) x degree)\n") ;
				return false ;
			}
			
			INT len = mxGetN(mx_arg);
			
			if (len ==  1)
				len = 0;
			
			return kernel->set_weights(mxGetPr(mx_arg), mxGetM(mx_arg), len);
			
		}
		
		// all other kernels
		CKernel *kernel = kernel_ ;
		INT num_subkernels = kernel->get_num_subkernels() ;
		if (mxGetM(mx_arg)!=1 || ((INT) mxGetN(mx_arg))!=num_subkernels)
		{
			SG_ERROR( "dimension mismatch (should be 1 x num_subkernels)\n") ;
			return false ;
		}
		
		kernel->set_subkernel_weights(mxGetPr(mx_arg), mxGetN(mx_arg));
		return true ;
	}

	SG_ERROR( "set_last_subkernel_weights only works for combined kernels") ;
	return false ;
}

bool CGUIMatlab::set_WD_position_weights(const mxArray* mx_arg)
{
	CKernel *kernel_ = gui->guikernel.get_kernel() ;

	if (kernel_ && (kernel_->get_kernel_type() == K_COMBINED))
		kernel_=((CCombinedKernel*)kernel_)->get_last_kernel() ;
	
	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
	{
		CWeightedDegreeStringKernel *kernel
			= (CWeightedDegreeStringKernel *) kernel_;
		if (mxGetM(mx_arg)!=1 & mxGetN(mx_arg)>0)
		{
			SG_ERROR( "dimension mismatch (should be 1xseq_length or 0x0)\n") ;
			return false ;
		}
		INT len = mxGetN(mx_arg);
		return kernel->set_position_weights(mxGetPr(mx_arg), len);
		
	}
	if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
	{
		CWeightedDegreePositionStringKernel *kernel
			= (CWeightedDegreePositionStringKernel *) kernel_;
		if (mxGetM(mx_arg)!=1 & mxGetN(mx_arg)>0)
		{
			SG_ERROR( "dimension mismatch (should be 1xseq_length or 0x0)\n") ;
			return false ;
		}
		if (mxGetM(mx_arg)==0 & mxGetN(mx_arg)==0)
			return kernel->delete_position_weights() ;
		else
		{
			INT len = mxGetN(mx_arg);
			return kernel->set_position_weights(mxGetPr(mx_arg), len);
		}
		
	}
	return false;
}

bool CGUIMatlab::set_WD_position_weights_per_example(const mxArray* mx_arg, const mxArray* mx_target)
{
	CKernel *kernel_ = gui->guikernel.get_kernel() ;

	if (kernel_ && (kernel_->get_kernel_type() == K_COMBINED))
		kernel_=((CCombinedKernel*)kernel_)->get_last_kernel() ;

	INT slen=0;
	CHAR* target=CGUIMatlab::get_mxString(mx_target, slen);
	if ( target && (!strncmp(target, "TRAIN", 5) || 
					!strncmp(target, "TEST", 4) ))
	{
		if (!strncmp(target, "TRAIN", 5))
		{
			if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
			{
				CWeightedDegreePositionStringKernel *kernel
					= (CWeightedDegreePositionStringKernel *) kernel_;
				if (mxGetM(mx_arg)==0 & mxGetN(mx_arg)==0)
					return kernel->delete_position_weights_lhs() ;
				else
				{
					INT len = mxGetM(mx_arg);
					INT num = mxGetN(mx_arg);
					return kernel->set_position_weights_lhs(mxGetPr(mx_arg), len, num);
				}
			}
		}
		else if (!strncmp(target, "TEST", 4))
		{
			if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
			{
				CWeightedDegreePositionStringKernel *kernel
					= (CWeightedDegreePositionStringKernel *) kernel_;
				if (mxGetM(mx_arg)==0 & mxGetN(mx_arg)==0)
					return kernel->delete_position_weights_rhs() ;
				else
				{
					INT len = mxGetM(mx_arg);
					INT num = mxGetN(mx_arg);
					return kernel->set_position_weights_rhs(mxGetPr(mx_arg), len, num);
				}
			}
		}
		delete[] target;
		return true ;
	} else
		return false ;
	
	
	return false;
}
#endif
