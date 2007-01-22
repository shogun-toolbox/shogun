/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Fabio De Bona
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#if defined(HAVE_R) && !defined(HAVE_SWIG)
#include <stdio.h>
#include <string.h>

#include "gui/TextGUI.h"
#include <R.h>
#include <Rinternals.h>

#include "guilib/GUIR.h"
#include "gui/GUI.h"

#include "lib/io.h"
#include "base/Version.h"
#include "distributions/hmm/HMM.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "kernel/WeightedDegreePositionCharKernel.h"
#include "kernel/CombinedKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "kernel/CustomKernel.h"
#include "kernel/LinearKernel.h"
#include "classifier/svm/SVM.h"

// This extern C declaration is needed to link
// shogun-R with the R interpreter (symbol differences)

extern "C"
{
	extern CTextGUI* gui;

	CGUI_R::CGUI_R() { }

	bool CGUI_R::send_command(CHAR* cmd)
	{
		return (gui->parse_line(cmd));
	}

	SEXP CGUI_R::get_hmm()
	{
		CHMM* h=gui->guihmm.get_current();

		if (h)
		{
			SEXP ans,p,q,a,b;

			/*
			 * transition matrix a 
			 * observation matrix b
			 * initial state distribution p
			 * accepting state distribution q
			 */

			PROTECT( p = allocVector(REALSXP, h->get_N() ));
			PROTECT( q = allocVector(REALSXP, h->get_N() ));

			PROTECT( a = allocMatrix(REALSXP, h->get_N(), h->get_N() ));
			PROTECT( b = allocMatrix(REALSXP, h->get_N(), h->get_M() ));

			int i,j;
			for (i=0; i<h->get_N(); i++)
			{
				REAL(p)[i]=h->get_p(i);
				REAL(q)[i]=h->get_q(i);
			}

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_N(); j++)
					REAL(a)[i+j*h->get_N()]=h->get_a(i,j);

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_M(); j++)
					REAL(b)[i+j*h->get_N()]=h->get_b(i,j);

			PROTECT( ans = allocList(0) );

			ans = CONS(p, ans);
			SET_TAG(ans, install("p"));

			ans = CONS(q, ans);
			SET_TAG(ans, install("q"));

			ans = CONS(a, ans);
			SET_TAG(ans, install("a"));

			ans = CONS(b, ans);
			SET_TAG(ans, install("b"));

			UNPROTECT(5);
			return ans;
		}

		return (R_NilValue);
	}
	
	SEXP CGUI_R::hmm_likelihood()
	{
		CHMM* h=gui->guihmm.get_current();

		if (h)
		{
			SEXP lik;
			PROTECT( lik = allocVector(REALSXP, 1 ));
			REAL(lik)[0]=h->model_probability();
			UNPROTECT(1);
			return lik;
		}
		return R_NilValue;
	}

	SEXP CGUI_R::best_path(int dim)
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
					SEXP ans, lik, path;

					PROTECT( lik = allocVector(REALSXP, 1 ));
					PROTECT( path = allocVector(REALSXP, num_feat ));

					T_STATES* s = h->get_path(dim, REAL(lik)[0]);

					for (int i=0; i<num_feat; i++)
						REAL(path)[i]=s[i];

					delete[] s;

					PROTECT( ans = allocList(0) );
					ans = CONS(lik, ans);
					SET_TAG(ans, install("likelihood"));
					ans = CONS(path, ans);
					SET_TAG(ans, install("path"));
					UNPROTECT(3);
					return ans;
				}
			}
		}

		return R_NilValue;
	}

	bool CGUI_R::append_hmm(const SEXP arg_list)
	{
		/*
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
			CHMM* h=new CHMM(N, M, NULL,
					gui->guihmm.get_pseudo(), gui->guihmm.get_number_of_tables());
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
						mxGetN(mx_b) == h->get_M() && mxGetM(mx_b) == h->get_N()
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
		*/
		return false;
	}

	bool CGUI_R::set_hmm(SEXP arg_list)
	{
		if( TYPEOF(arg_list) != LISTSXP ) {
			SG_ERROR( "You have to supply an argument pairlist of length four.\n");
			return false;
		}

		SEXP p,q,a,b;

		b = CAR(arg_list);
		arg_list = CDR(arg_list);
		a = CAR(arg_list);
		arg_list = CDR(arg_list);
		q = CAR(arg_list);
		arg_list = CDR(arg_list);
		p = CAR(arg_list);
		
		INT N=Rf_nrows(a);
		INT M=Rf_ncols(b);

		CHMM* h=new CHMM(N, M, NULL, gui->guihmm.get_pseudo());

		if (h)
		{

			if (
					Rf_nrows(p) == h->get_N() && Rf_ncols(p) == 1 &&
					Rf_nrows(q) == h->get_N() && Rf_ncols(q) == 1 &&
					Rf_nrows(a) == h->get_N() && Rf_ncols(a) == h->get_N() &&
					Rf_nrows(b) == h->get_N() && Rf_ncols(b) == h->get_M()
			   )
			{
				int i,j;
				for (i=0; i< h->get_N(); i++)
				{
					h->set_p(i, REAL(p)[i]);
					h->set_q(i, REAL(q)[i]);
				}

				for (i=0; i< h->get_N(); i++)
					for (j=0; j< h->get_N(); j++)
						h->set_a(i,j, REAL(a)[i+j*h->get_N()]);

				for (i=0; i< h->get_N(); i++)
					for (j=0; j< h->get_M(); j++)
						h->set_b(i,j, REAL(b)[i+j*h->get_N()]);

				gui->guihmm.set_current(h);
				return true;
			}
			else
			{
				SG_ERROR( "model matricies not matching in size\n");
				SG_ERROR( "N:%d M:%d p:(%d,%d) q:(%d,%d) a:(%d,%d) b(%d,%d)\n",
						N, M,
						Rf_nrows(p), Rf_ncols(p), 
						Rf_nrows(q), Rf_ncols(q), 
						Rf_nrows(a), Rf_ncols(a), 
						Rf_nrows(b), Rf_ncols(b));
			}
		}
		return false;
	}

	SEXP CGUI_R::hmm_classify_example(int idx)
	{
		/*
		mxArray* mx_result=mxCreateDoubleMatrix(1, 1, mxDREAL);
		double* result=mxGetPr(mx_result);
		*result=gui->guihmm.classify_example(idx);
		retvals[0]=mx_result;*/
		return R_NilValue;
	}

	SEXP CGUI_R::one_class_hmm_classify()
	{
		/*
		CFeatures* f=gui->guifeatures.get_test_features();
		if (f)
		{
			int num_vec=f->get_num_vectors();

			mxArray* mx_result = mxCreateDoubleMatrix(1, num_vec, mxDREAL);
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
		*/
		return R_NilValue;
	}

	SEXP CGUI_R::one_class_hmm_classify_example(int idx)
	{
		/*
		mxArray* mx_result=mxCreateDoubleMatrix(1, 1, mxDREAL);
		double* result=mxGetPr(mx_result);
		*result=gui->guihmm.one_class_classify_example(idx);
		retvals[0]=mx_result;
		*/
		return R_NilValue;
	}

	SEXP CGUI_R::hmm_classify()
	{
		CFeatures* f=gui->guifeatures.get_test_features();
		if (f)
		{
			int num_vec=f->get_num_vectors();
			SEXP ans;

			PROTECT( ans = allocVector(REALSXP, num_vec ));
			CLabels* l=gui->guihmm.classify();

			for (int i=0; i<num_vec; i++)
				REAL(ans)[i]=l->get_label(i);

			delete l;
			UNPROTECT(1);

			return ans;
		}
		return (R_NilValue);
	}

	/* This method returns a List expression
	 * including the index of the support vectors
	 * the weights (alphas) and the bias value b. 
	 */

	SEXP CGUI_R::get_svm()
	{
		CSVM* svm=gui->guisvm.get_svm();

		if (svm)
		{
			SG_DEBUG("Acquired svm.");
			SEXP ans, alphas, b, SV;
			int numSV = svm->get_num_support_vectors();

			PROTECT( alphas = allocVector(REALSXP, svm->get_num_support_vectors() ));
			PROTECT( b = allocVector(REALSXP, 1));
			PROTECT( SV = allocVector(INTSXP, numSV ));

			REAL(b)[0] = svm->get_bias();

			for (int i=0; i<numSV; i++)
			{
				double t = svm->get_alpha(i);
				REAL(alphas)[i]= t;
				int sv_num = svm->get_support_vector(i);
				INTEGER(SV)[i]= sv_num ;
			}

			PROTECT( ans = allocList(0) );

			ans = CONS(SV, ans);
			SET_TAG(ans, install("SV"));

			ans = CONS(b, ans);
			SET_TAG(ans, install("b"));

			ans = CONS(alphas, ans);
			SET_TAG(ans, install("alphas"));

			UNPROTECT(4);
			return ans;
		}

		return(R_NilValue);
	}


	SEXP CGUI_R::svm_classify()
	{
		CFeatures* f=gui->guifeatures.get_test_features();
		if (f)
		{
			SEXP output;
			int num_vec=f->get_num_vectors();
			PROTECT( output = allocMatrix(REALSXP, 1, num_vec) );

			CLabels* l=gui->guisvm.classify();

			if (!l)
			{
				SG_ERROR( "svm_classify failed\n") ;
				return false ;
			} ;

			for (int i=0; i<num_vec; i++)
				REAL(output)[i]=l->get_label(i);
			delete l;

			UNPROTECT(1) ;

			return output;
		}
		return R_NilValue;
	}


	bool CGUI_R::set_svm(SEXP arg_list)
	{
		if( TYPEOF(arg_list) != LISTSXP ) {
			SG_ERROR( "You have to supply an argument list of length four.\n");
			return false;
		}

		CSVM* svm=gui->guisvm.get_svm();

		if (svm)
		{
		SEXP b,alphas;

		b = CAR(arg_list);
		arg_list = CDR(arg_list);
		alphas = CAR(arg_list);

			if (
					Rf_nrows(b)==1  && Rf_ncols(b) == 1 &&
					Rf_ncols(alphas) == 2 && Rf_nrows(alphas)>0
			   )
			{
				svm->create_new_model(Rf_nrows(alphas));
				svm->set_bias(REAL(b)[0]);

				for (int i=0; i< svm->get_num_support_vectors(); i++)
				{
					svm->set_alpha(i, REAL(alphas)[i]);
					svm->set_support_vector(i, (int) REAL(alphas)[i+svm->get_num_support_vectors()]);
				}

				return true;
			}
		}
		else
			SG_ERROR( "no svm object available\n") ;

		return false;
	}

	SEXP CGUI_R::svm_classify_example(int idx)
	{
		/*
		mxArray* mx_result=mxCreateDoubleMatrix(1, 1, mxDREAL);
		retvals[0]=mx_result;
		double* result=mxGetPr(mx_result);

		if (!gui->guisvm.classify_example(idx, result[0]))
		{
			SG_ERROR( "svm_classify_example failed\n") ;
			return false ;
		} ;
*/
		return R_NilValue;
	}

	SEXP CGUI_R::get_features(CFeatures* f)
	{
		if (f)
		{
			SEXP feat=NULL;

			switch (f->get_feature_class())
			{
				case C_SIMPLE:
					switch (f->get_feature_type())
					{
						case F_DREAL:
							{

								int rows = ((CRealFeatures*) f)->get_num_features();
								int cols = ((CRealFeatures*) f)->get_num_vectors();

								PROTECT( feat = allocMatrix(REALSXP, rows, cols) );

								for (INT i=0; i<((CRealFeatures*) f)->get_num_vectors(); i++)
								{
									INT num_feat;
									bool free_vec;
									DREAL* vec=((CRealFeatures*) f)->get_feature_vector(i, num_feat, free_vec);

									for (INT j=0; j<num_feat; j++) {
										REAL(feat)[rows*i+j] = (double) vec[j];
									}
									((CRealFeatures*) f)->free_feature_vector(vec, i, free_vec);
								}

								UNPROTECT(1) ;
								break;
							}
							/*
							   case F_WORD:

							   if (mx_feat)
							   {
							   WORD* feat=(WORD*) mxGetData(mx_feat);

							   for (INT i=0; i<((CWordFeatures*) f)->get_num_vectors(); i++)
							   {
							   INT num_feat;
							   bool free_vec;
							   WORD* vec=((CWordFeatures*) f)->get_feature_vector(i, num_feat, free_vec);
							   for (INT j=0; j<num_feat; j++)
							   feat[((CWordFeatures*) f)->get_num_vectors()*j+i]= vec[j];
							   ((CWordFeatures*) f)->free_feature_vector(vec, i, free_vec);
							   }
							   }
							   break;
							   case F_SHORT:
							   mx_feat=mxCreateNumericMatrix(((CShortFeatures*) f)->get_num_vectors(), ((CShortFeatures*) f)->get_num_features(), mxINT16_CLASS, mxDREAL);

							   if (mx_feat)
							   {
							   SHORT* feat=(SHORT*) mxGetData(mx_feat);

							   for (INT i=0; i<((CShortFeatures*) f)->get_num_vectors(); i++)
							   {
							   INT num_feat;
							   bool free_vec;
							   SHORT* vec=((CShortFeatures*) f)->get_feature_vector(i, num_feat, free_vec);
							   for (INT j=0; j<num_feat; j++)
							   feat[((CShortFeatures*) f)->get_num_vectors()*j+i]= vec[j];
							   ((CShortFeatures*) f)->free_feature_vector(vec, i, free_vec);
							   }
							   }
							   break;
							   case F_CHAR:
							   mx_feat=mxCreateNumericMatrix(((CCharFeatures*) f)->get_num_vectors(), ((CCharFeatures*) f)->get_num_features(), mxCHAR_CLASS, mxDREAL);

							   if (mx_feat)
							   {
							   CHAR* feat=(CHAR*) mxGetData(mx_feat);

							   for (INT i=0; i<((CCharFeatures*) f)->get_num_vectors(); i++)
							   {
							   INT num_feat;
							   bool free_vec;
							   CHAR* vec=((CCharFeatures*) f)->get_feature_vector(i, num_feat, free_vec);
							   for (INT j=0; j<num_feat; j++)
							   feat[((CCharFeatures*) f)->get_num_vectors()*j+i]= vec[j];
							   ((CCharFeatures*) f)->free_feature_vector(vec, i, free_vec);
							   }
							   }
							   break;
							   case F_BYTE:
							   mx_feat=mxCreateNumericMatrix(((CByteFeatures*) f)->get_num_vectors(), ((CByteFeatures*) f)->get_num_features(), mxUINT8_CLASS, mxDREAL);

							   if (mx_feat)
							   {
							   BYTE* feat=(BYTE*) mxGetData(mx_feat);

							   for (INT i=0; i<((CByteFeatures*) f)->get_num_vectors(); i++)
							   {
							   INT num_feat;
							   bool free_vec;
							   BYTE* vec=((CByteFeatures*) f)->get_feature_vector(i, num_feat, free_vec);
							   for (INT j=0; j<num_feat; j++)
							   feat[((CByteFeatures*) f)->get_num_vectors()*j+i]= vec[j];
							   ((CByteFeatures*) f)->free_feature_vector(vec, i, free_vec);
							   }
							   }
							break;
							*/
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
				case C_STRING:
					switch (f->get_feature_type())
					{
						case F_CHAR:
							{
								int num_vec=f->get_num_vectors();
								PROTECT( feat = allocVector(STRSXP, num_vec) );
								for (int i=0; i<num_vec; i++)
								{
									INT len=0;
									CHAR* fv=((CStringFeatures<CHAR>*) f)->get_feature_vector(i, len);

									if (len>0)
									{
										char* str=new char[len+1];
										strncpy(str, fv, len);
										str[len]='\0';
										SG_DEBUG("str[%d]=%s\n", i, str);
										SET_STRING_ELT(feat, i, mkChar(str));
										delete[] str;
									}
									else
										SET_STRING_ELT(feat, i, mkChar(""));
								}
								UNPROTECT(1) ;
							}
							break;
						default:
							io.not_implemented();
					};

					break;
				default:
					io.not_implemented();
			}

			return feat;
		}

		return R_NilValue;
	}

#if 0
   
//bool CGUI_R::set_custom_kernel(const mxArray* vals[], bool source_is_diag, bool dest_is_diag) {
bool CGUI_R::set_custom_kernel(SEXP args) {

	* action=REAL(VECTOR_ELT(CAR(args), 0));
	args = CDR(args); /* pop action out of list */

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

		return R_NilValue;
	}

#endif

	CFeatures* CGUI_R::set_features(SEXP feat, SEXP alphabet)
	{
		CFeatures* f=NULL;
		SG_INFO( "start CGUI_R::set_features\n") ;

		if (feat)
		{
			if (false)
				io.not_implemented();
			else
			{
				if( TYPEOF(feat) == REALSXP || TYPEOF(feat) == INTSXP )
				{
					int rows = Rf_nrows(feat);
					int cols = Rf_ncols(feat);

					f= new CRealFeatures(0);

					INT num_vec=cols;
					INT num_feat=rows;
					DREAL* fm=new DREAL[num_vec*num_feat];
					ASSERT(fm);

					for (INT i=0; i<num_vec; i++)
						for (INT j=0; j<num_feat; j++)
							fm[i*num_feat+j]= REAL(feat)[i*num_feat+j];

					((CRealFeatures*) f)->set_feature_matrix(fm, num_feat, num_vec);
				}
				else if (TYPEOF(feat) == STRSXP )
				{
					if (alphabet && TYPEOF(alphabet) == STRSXP)
					{
						int num_vec = length(feat);
						CHAR* al=CHAR(VECTOR_ELT(alphabet, 0));
						CAlphabet* alpha = new CAlphabet(al, strlen(al));
						T_STRING<CHAR>* sc=new T_STRING<CHAR>[num_vec];
						ASSERT(alpha);
						ASSERT(sc);

						int maxlen=0;
						alpha->clear_histogram();

						for (int i=0; i<num_vec; i++)
						{
							SEXPREC* s= STRING_ELT(feat,i);
							CHAR* c= CHAR(s);
							int len=LENGTH(s);

							if (len && c)
							{
								CHAR* dst=new CHAR[len];
								sc[i].string=(CHAR*) memcpy(dst, c, len*sizeof(CHAR));
								sc[i].length=len;
								maxlen=CMath::max(maxlen, len);
								alpha->add_string_to_histogram(sc[i].string, sc[i].length);
							}
							else
							{
								SG_WARNING( "string with index %d has zero length\n", i+1);
								sc[i].string=0;
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
				}
			}
		}
		return f;
	}

	SEXP CGUI_R::get_version()
	{
		SEXP ans;
		PROTECT( ans = allocVector(REALSXP, 1));
		REAL(ans)[0] = version.get_version_revision();
		UNPROTECT(1);
		return ans;
	}

   
SEXP CGUI_R::get_svm_objective() {
   SEXP ans;
	//mxArray* mx_v=mxCreateDoubleMatrix(1, 1, mxDREAL);
	PROTECT( ans = allocVector(REALSXP, 1));
	CSVM* svm=gui->guisvm.get_svm();

	if (svm)
	{
		//double* v=mxGetPr(mx_v);
		//*v = svm->get_objective();
      REAL(ans)[0] = svm->get_objective();

		//retvals[0]=mx_v;
      UNPROTECT(1);
	   return ans;
	}
	else
	   SG_ERROR( "no svm set\n");

	return R_NilValue;
	}

	SEXP CGUI_R::get_labels(CLabels* label)
	{
		SEXP lab ;

		if (label)
		{
			PROTECT( lab = allocVector(INTSXP, label->get_num_labels()) );

			for (int i=0; i< label->get_num_labels(); i++)
				INTEGER(lab)[i]=label->get_int_label(i);

			UNPROTECT(1) ;
			return lab ;
		}
		return R_NilValue;
	}


	CLabels* CGUI_R::set_labels(SEXP labelsR)
	{
		if(labelsR && TYPEOF(labelsR)==REALSXP)
		{
			SG_DEBUG("lenght of labels is %d", length(labelsR));
			CLabels* label=new CLabels(length(labelsR));

			double* lab= REAL(labelsR);

			SG_INFO( "%d\n", label->get_num_labels());

			for (int i=0; i<label->get_num_labels(); i++)
				if (!label->set_label(i, lab[i]))
					SG_ERROR( "weirdo ! %d %d\n", label->get_num_labels(), i);

			return label;
		}

		return NULL;
	}

	/*
	 * This function returns the kernel matrix.
	 *
	 *
	 *
	 */

	SEXP CGUI_R::get_kernel_matrix()
	{
		CKernel* k = gui->guikernel.get_kernel();

		if (k && k->get_rhs() && k->get_lhs())
		{
			int num_vec1=k->get_lhs()->get_num_vectors();
			int num_vec2=k->get_rhs()->get_num_vectors();
			SG_DEBUG("Kernel matrix has size %d / %d\n", num_vec1, num_vec2);
			SEXP ans,dim;
			PROTECT( ans = allocMatrix(REALSXP, num_vec1, num_vec2) );

			DREAL* blub = k->get_kernel_matrix_real(num_vec1, num_vec2,NULL);

			if( blub == NULL )
				SG_DEBUG("error return value is NULL!");

			for( int i=0; i<num_vec1; i++) 
			{
				for( int j=0; j<num_vec2; j++) 
				{
					REAL(ans)[i+num_vec2*j] = blub[i+num_vec2*j];
				}
			}

			PROTECT( dim = allocVector(INTSXP, 2) );
			INTEGER(dim)[0] = num_vec1;
			INTEGER(dim)[1] = num_vec2;
			setAttrib(ans, R_DimSymbol, dim);

			UNPROTECT(2);
			SG_DEBUG("matrix created!");
			return ans;
		}
		else
			SG_ERROR( "no kernel defined");

		return(R_NilValue);
	}

#if 0
	bool CGUI_R::get_kernel_optimization(mxArray* retvals[])
	{
		CKernel *kernel_ = gui->guikernel.get_kernel() ;

		if (kernel_)
		{
			switch (kernel_->get_kernel_type())
			{
				case K_WEIGHTEDDEGREE:
					{
						CWeightedDegreeCharKernel *kernel = (CWeightedDegreeCharKernel *) kernel_ ;

						if (kernel->get_max_mismatch()!=0)
							return false ;

						INT len=0 ;
						DREAL* res=kernel->compute_abs_weights(len) ;

						mxArray* mx_result=mxCreateDoubleMatrix(4, len, mxDREAL);
						double* result=mxGetPr(mx_result);
						for (int i=0; i<4*len; i++)
							result[i]=res[i] ;
						delete[] res ;

						retvals[0]=mx_result;
						return true;
					}
				case K_WEIGHTEDDEGREEPOS:
					{
						CWeightedDegreePositionCharKernel *kernel = (CWeightedDegreePositionCharKernel *) kernel_ ;

						if (kernel->get_max_mismatch()!=0)
							return false ;

						INT len=0 ;
						DREAL* res=kernel->compute_abs_weights(len) ;

						mxArray* mx_result=mxCreateDoubleMatrix(4, len, mxDREAL);
						double* result=mxGetPr(mx_result);
						for (int i=0; i<4*len; i++)
							result[i]=res[i] ;
						delete[] res ;

						retvals[0]=mx_result;
						return true;
					}
				case  K_COMMWORDSTRING:
					{
						CCommWordStringKernel *kernel = (CCommWordStringKernel *) kernel_ ;

						INT len=0 ;
						DREAL* weights ;
						kernel->get_dictionary(len, weights) ;

						mxArray* mx_result=mxCreateDoubleMatrix(len, 1, mxDREAL);
						double* result=mxGetPr(mx_result);
						for (int i=0; i<len; i++)
							result[i]=weights[i] ;

						retvals[0]=mx_result;
						return true;
					}
				case  K_LINEAR:
					{
						CLinearKernel *kernel = (CLinearKernel *) kernel_ ;

						INT len=0 ;
						const double* weights = kernel->get_normal(len);

						mxArray* mx_result=mxCreateDoubleMatrix(len, 1, mxDREAL);
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

	bool CGUI_R::compute_by_subkernels(mxArray* retvals[])
	{
		CKernel *kernel_ = gui->guikernel.get_kernel() ;

		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
		{
			CWeightedDegreeCharKernel *kernel = (CWeightedDegreeCharKernel *) kernel_ ;

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

			mxArray* mx_result=mxCreateDoubleMatrix(degree*len, num, mxDREAL);
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
			CWeightedDegreePositionCharKernel *kernel = (CWeightedDegreePositionCharKernel *) kernel_ ;

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

			mxArray* mx_result=mxCreateDoubleMatrix(degree*len, num, mxDREAL);
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

#endif

	/*
	 * When performing multiple kernel learning
	 * this function return the weights of the
	 * respective kernels after training.
	 *
	 *
	 */

	SEXP CGUI_R::get_subkernel_weights()
	{
		CKernel *kernel_ = gui->guikernel.get_kernel() ;
		INT degree=-1;
		INT length=-1;

		SEXP result;

		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
		{
			CWeightedDegreeCharKernel *kernel = (CWeightedDegreeCharKernel *) kernel_ ;

			//SG_DEBUG("getting degree weights...");
			const DREAL* weights = kernel->get_degree_weights(degree, length) ;
			if (length == 0)
				length = 1;

			PROTECT( result = allocVector(REALSXP, degree*length  ));

			for (int i=0; i<degree*length; i++)
				REAL(result)[i] = weights[i] ;

			UNPROTECT(1);

			return result;
		}

		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
		{
			CWeightedDegreePositionCharKernel *kernel = (CWeightedDegreePositionCharKernel *) kernel_ ;

			const DREAL* weights = kernel->get_degree_weights(degree, length) ;
			if (length == 0)
				length = 1;

			PROTECT( result = allocVector(REALSXP, degree*length ));

			for (int i=0; i<degree*length; i++)
				REAL(result)[i] = weights[i] ;

			UNPROTECT(1);

			return result;
		}

		if (kernel_ && (kernel_->get_kernel_type() == K_COMBINED))
		{
			//SG_DEBUG("getting combined weights...");
			CCombinedKernel *kernel = (CCombinedKernel *) kernel_ ;
			INT num_weights = -1 ;
			const DREAL* weights = kernel->get_subkernel_weights(num_weights) ;

			PROTECT( result = allocVector(REALSXP, num_weights ));

			for (int i=0; i<num_weights; i++)
				REAL(result)[i] = weights[i] ;

			UNPROTECT(1);

			return result;
		}

		//SG_DEBUG("getting no weights...");
		return (R_NilValue);
	}

#if 0
	bool CGUI_R::get_last_subkernel_weights(mxArray* retvals[])
	{
		CKernel *ckernel = gui->guikernel.get_kernel() ;
		if (ckernel && (ckernel->get_kernel_type() == K_COMBINED))
		{
			CKernel *kernel_ = ((CCombinedKernel*)ckernel)->get_last_kernel() ;

			INT degree=-1;
			INT length=-1;

			if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
			{
				CWeightedDegreeCharKernel *kernel = (CWeightedDegreeCharKernel *) kernel_ ;

				const DREAL* weights = kernel->get_degree_weights(degree, length) ;
				if (length == 0)
					length = 1;

				mxArray* mx_result=mxCreateDoubleMatrix(degree, length, mxDREAL);
				double* result=mxGetPr(mx_result);

				for (int i=0; i<degree*length; i++)
					result[i] = weights[i] ;

				retvals[0]=mx_result;
				return true;
			}

			if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
			{
				CWeightedDegreePositionCharKernel *kernel = (CWeightedDegreePositionCharKernel *) kernel_ ;

				const DREAL* weights = kernel->get_degree_weights(degree, length) ;
				if (length == 0)
					length = 1;

				mxArray* mx_result=mxCreateDoubleMatrix(degree, length, mxDREAL);
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

				mxArray* mx_result=mxCreateDoubleMatrix(1, num_weights, mxDREAL);
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

	bool CGUI_R::get_WD_position_weights(mxArray* retvals[])
	{
		CKernel *kernel_ = gui->guikernel.get_kernel() ;
		INT length=-1;

		if (kernel_ && (kernel_->get_kernel_type() == K_COMBINED))
			kernel_=((CCombinedKernel*)kernel_)->get_last_kernel() ;

		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
		{
			CWeightedDegreeCharKernel *kernel = (CWeightedDegreeCharKernel *) kernel_ ;

			const DREAL* position_weights = kernel->get_position_weights(length) ;
			mxArray* mx_result ;
			if (position_weights==NULL)
				mx_result=mxCreateDoubleMatrix(1, 0, mxDREAL);
			else
			{
				mx_result=mxCreateDoubleMatrix(1, length, mxDREAL);
				double* result=mxGetPr(mx_result);

				for (int i=0; i<length; i++)
					result[i] = position_weights[i] ;
			}
			retvals[0]=mx_result;
			return true;
		}
		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREEPOS))
		{
			CWeightedDegreePositionCharKernel *kernel = (CWeightedDegreePositionCharKernel *) kernel_ ;

			const DREAL* position_weights = kernel->get_position_weights(length) ;
			mxArray* mx_result ;
			if (position_weights==NULL)
				mx_result=mxCreateDoubleMatrix(1, 0, mxDREAL);
			else
			{
				mx_result=mxCreateDoubleMatrix(1, length, mxDREAL);
				double* result=mxGetPr(mx_result);

				for (int i=0; i<length; i++)
					result[i] = position_weights[i] ;
			}
			retvals[0]=mx_result;
			return true;
		}
		return false;
	}

	bool CGUI_R::set_subkernel_weights(const mxArray* mx_arg)
	{
		CKernel *kernel_ = gui->guikernel.get_kernel() ;

		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
		{
			CWeightedDegreeCharKernel *kernel = (CWeightedDegreeCharKernel *) kernel_ ;
			INT degree = kernel->get_degree() ;
			if (mxGetM(mx_arg)!=degree || mxGetN(mx_arg)<1)
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
			CWeightedDegreePositionCharKernel *kernel = (CWeightedDegreePositionCharKernel *) kernel_ ;
			INT degree = kernel->get_degree() ;
			if (mxGetM(mx_arg)!=degree || mxGetN(mx_arg)<1)
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
		if (mxGetM(mx_arg)!=1 || mxGetN(mx_arg)!=num_subkernels)
		{
			SG_ERROR( "dimension mismatch (should be 1 x num_subkernels)\n") ;
			return false ;
		}

		kernel->set_subkernel_weights(mxGetPr(mx_arg), mxGetN(mx_arg));
		return true ;
	}

	bool CGUI_R::set_last_subkernel_weights(const mxArray* mx_arg)
	{
		CKernel *ckernel = gui->guikernel.get_kernel() ;
		if (ckernel && (ckernel->get_kernel_type() == K_COMBINED))
		{
			CKernel *kernel_ = ((CCombinedKernel*)ckernel)->get_last_kernel() ;

			if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
			{
				CWeightedDegreeCharKernel *kernel = (CWeightedDegreeCharKernel *) kernel_ ;
				INT degree = kernel->get_degree() ;
				if (mxGetM(mx_arg)!=degree || mxGetN(mx_arg)<1)
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
				CWeightedDegreePositionCharKernel *kernel = (CWeightedDegreePositionCharKernel *) kernel_ ;
				INT degree = kernel->get_degree() ;
				if (mxGetM(mx_arg)!=degree || mxGetN(mx_arg)<1)
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
			if (mxGetM(mx_arg)!=1 || mxGetN(mx_arg)!=num_subkernels)
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

	bool CGUI_R::set_WD_position_weights(const mxArray* mx_arg)
	{
		CKernel *kernel_ = gui->guikernel.get_kernel() ;

		if (kernel_ && (kernel_->get_kernel_type() == K_COMBINED))
			kernel_=((CCombinedKernel*)kernel_)->get_last_kernel() ;

		if (kernel_ && (kernel_->get_kernel_type() == K_WEIGHTEDDEGREE))
		{
			CWeightedDegreeCharKernel *kernel = (CWeightedDegreeCharKernel *) kernel_ ;
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
			CWeightedDegreePositionCharKernel *kernel = (CWeightedDegreePositionCharKernel *) kernel_ ;
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

#endif

} // end of extern "C"

#endif // HAVE_R
