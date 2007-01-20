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

#ifndef HAVE_SWIG

#include "guilib/GUIFeatures.h"
#include "gui/GUI.h"
#include "lib/io.h"

	CGUIFeatures::CGUIFeatures(CGUI * gui_)
: gui(gui_), train_features(NULL), test_features(NULL), ref_features(NULL)
{
}

CGUIFeatures::~CGUIFeatures()
{
	delete train_features;
	delete test_features;
	delete ref_features;
}

void CGUIFeatures::invalidate_train()
{
	CKernel *k = gui->guikernel.get_kernel() ;
	if (k)
		k->remove_lhs() ;
}

void CGUIFeatures::invalidate_test()
{
	CKernel *k = gui->guikernel.get_kernel() ;
	if (k)
		k->remove_rhs() ;
}

bool CGUIFeatures::load(CHAR* param)
{
	param=CIO::skip_spaces(param);
	CHAR filename[1024]="";
	CHAR target[1024]="";
	CHAR type[1024]="";
	CHAR fclass[1024]="";
	bool result=false;
	INT comp_features=0;
	INT size=100;

	if ((sscanf(param, "%s %s %s %s %d  %d", filename, fclass, type, target, &size, &comp_features))>=4)
	{
		CFeatures** f_ptr=NULL;

		if (strcmp(target,"TRAIN")==0)
		{
			f_ptr=&train_features;
			invalidate_train() ;
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_features;
			invalidate_test() ;
		}
		else
		{
			SG_ERROR( "see help for parameters\n");
			return false;
		}

		delete (*f_ptr);
		*f_ptr=NULL;

		if (strcmp(fclass, "SIMPLE")==0)
		{
			if (strcmp(type,"REAL")==0)
			{
				*f_ptr=new CRealFeatures(filename);
				//SG_DEBUG( "opening file...\n");
				//*f_ptr=new CRealFileFeatures(size, filename);

				//if (comp_features)
				//((CRealFileFeatures*) *f_ptr)->load_feature_matrix() ;
			}
			else if (strcmp(type, "BYTE")==0)
			{
				///FIXME make CHAR type configurable... it is DNA by default
				*f_ptr=new CByteFeatures(DNA, filename);
			}
			else if (strcmp(type, "CHAR")==0)
			{
				///FIXME make CHAR type configurable... it is DNA by default
				*f_ptr=new CCharFeatures(DNA, filename);
			}
			else if (strcmp(type, "SHORT")==0)
			{
				*f_ptr=new CShortFeatures(filename);
			}
			else
			{
				SG_ERROR( "unknown type\n");
				return false;
			}
		}
		else if (strcmp(fclass, "SPARSE")==0)
		{
			io.not_implemented();
		}
		else if (strcmp(fclass, "STRING")==0)
		{
			if (strcmp(type,"REAL")==0)
			{
				*f_ptr=new CStringFeatures<DREAL>(filename);
			}
			else if (strcmp(type, "BYTE")==0)
			{
				///FIXME make CHAR type configurable... it is DNA by default
				*f_ptr=new CStringFeatures<BYTE>(filename, DNA);
			}
			else if (strcmp(type, "CHAR")==0)
			{
				///FIXME make CHAR type configurable... it is DNA by default
				*f_ptr=new CStringFeatures<CHAR>(filename, DNA);
			}
			else if (strcmp(type, "SHORT")==0)
			{
				*f_ptr=new CStringFeatures<SHORT>(filename);
			}
			else if (strcmp(type, "WORD")==0)
			{
				*f_ptr=new CStringFeatures<WORD>(filename);
			}
			else if (strcmp(type, "ULONG")==0)
			{
				*f_ptr=new CStringFeatures<ULONG>(filename);
			}
			else
			{
				SG_ERROR( "unknown type\n");
				return false;
			}
		}

	} else
		SG_ERROR( "see help for params\n");

	return result;
}

bool CGUIFeatures::save(CHAR* param)
{
	bool result=false;
	param=CIO::skip_spaces(param);
	CHAR fname[1024]="";
	CHAR target[1024]="";
	CHAR type[1024]="";

	if ((sscanf(param, "%s %s %s", fname, type, target))==3)
	{

		CFeatures** f_ptr=NULL;

		if (strcmp(target,"TRAIN")==0)
		{
			f_ptr=&train_features;
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_features;
		}
		else
		{
			SG_ERROR( "see help for parameters\n");
			return false;
		}

		if (*f_ptr)
		{
			if (fname)
			{
				if (strcmp(type,"REAL")==0)
				{
					result= ((CRealFeatures*) (*f_ptr))->save(fname);
				}
				else if (strcmp(type, "BYTE")==0)
				{
					result= ((CByteFeatures*) (*f_ptr))->save(fname);
				}
				else if (strcmp(type, "CHAR")==0)
				{
					result= ((CCharFeatures*) (*f_ptr))->save(fname);
				}
				else if (strcmp(type, "SHORT")==0)
				{
					result= ((CShortFeatures*) (*f_ptr))->save(fname);
				}
				else if (strcmp(type, "WORD")==0)
				{
					result= ((CWordFeatures*) (*f_ptr))->save(fname);
				}
				else
				{
					SG_ERROR( "unknown type\n");
					return false;
				}
			}

			if (!result)
				SG_ERROR( "writing to file %s failed!\n", fname);
			else
			{
				SG_INFO( "successfully written features into \"%s\" !\n", fname);
				result=true;
			}

		} else
			SG_ERROR( "set features first\n");

	} else
		SG_ERROR( "see help for params\n");

	return result;
}

bool CGUIFeatures::clean(CHAR* param)
{
	param=CIO::skip_spaces(param);
	CHAR target[1024]="";

	if ((sscanf(param, "%s", target))==1)
	{
		if (strcmp(target,"TRAIN")==0)
			set_train_features(NULL);
		else if (strcmp(target,"TEST")==0)
			set_test_features(NULL);
		else
		{
			SG_ERROR( "see help for parameters\n");
			return false;
		}
		return true;

	} else
		SG_ERROR( "see help for params\n");

	return false ;
}

bool CGUIFeatures::reshape(CHAR* param)
{
	bool result=false;
	INT num_feat=0;
	INT num_vec=0;
	CHAR target[1024]="";

	CFeatures** f_ptr=NULL;

	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %d %d", target, &num_feat, &num_vec))==3)
	{
		if (strcmp(target,"TRAIN")==0)
		{
			f_ptr=&train_features;
			invalidate_train() ;
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_features;
			invalidate_test() ;
		}
	}
	else
		SG_ERROR( "see help for params\n");

	if (f_ptr)
	{
		SG_INFO( "reshape data to %d x %d\n", num_feat, num_vec);
		result=(*f_ptr)->reshape(num_feat, num_vec);

		if (!result)
			SG_ERROR( "reshaping failed");
	}

	return result;
}

CSparseFeatures<DREAL>* CGUIFeatures::convert_simple_real_to_sparse_real(CRealFeatures* src, CHAR* param)
{
	if (src)
	{
		if ( (src->get_feature_class()) == C_SIMPLE)
		{
			if ( (src->get_feature_type()) == F_DREAL)
			{
				//create sparse features with 0 cache
				SG_INFO( "attempting to convert dense feature matrix to a sparse one\n");
				CSparseFeatures<DREAL>* target=new CSparseFeatures<DREAL>(0);
				INT num_f=0;
				INT num_v=0;
				DREAL* feats=src->get_feature_matrix(num_f, num_v);
				if (target->set_full_feature_matrix(feats, num_f, num_v))
					return target;

				delete target;
			}
		}
	}

	return NULL;
}

CStringFeatures<CHAR>* CGUIFeatures::convert_simple_char_to_string_char(CCharFeatures* src, CHAR* param)
{
	INT num_vec=src->get_num_vectors();
	T_STRING<CHAR>* strings=new T_STRING<CHAR>[num_vec];
	INT max_len=-1;

	for (INT i=0; i<num_vec; i++)
	{
		bool to_free=false;
		INT len=0;
		CHAR* str= src->get_feature_vector(i, len, to_free);
		strings[i].length=len ;
		for (int j=0; j<len; j++)
			if (str[j]==0)
			{
				strings[i].length=j ;
				break ;
			} ;
		strings[i].string=new CHAR[strings[i].length];

		for (int j=0; j<strings[i].length; j++)
			strings[i].string[j]=str[j];

		if (strings[i].length> max_len)
			max_len=strings[i].length;

		src->free_feature_vector(str, i, to_free);
	}

	CStringFeatures<CHAR>* target= new CStringFeatures<CHAR>(new CAlphabet(src->get_alphabet()));
	target->set_features(strings, num_vec, max_len);

	return target;
}

CRealFeatures* CGUIFeatures::convert_simple_word_to_simple_salzberg(CWordFeatures* src, CHAR* param)
{
	CPluginEstimate* pie=gui->guipluginestimate.get_estimator();
	ASSERT(src->get_feature_type()==F_WORD && src->get_feature_class()==C_SIMPLE);
	ASSERT(pie);

	CRealFeatures* target=new CRealFeatures(0);
	ASSERT(target);

	INT num_feat=src->get_num_features();
	INT num_vec=src->get_num_vectors();
	DREAL* fm=new DREAL[num_vec*num_feat];

	if (fm)
	{
		for (INT i=0; i<num_vec; i++)
		{
			INT len=0;
			bool to_free=false;
			WORD* vec = src->get_feature_vector(i, len, to_free);
			ASSERT(num_feat==len);

			for (INT j=0; j<num_feat; j++)
				fm[i*num_feat+j]=pie->get_parameterwise_log_odds(vec[j], j);

			src->free_feature_vector(vec, i, to_free);
		}
		target->set_feature_matrix(fm, num_feat, num_vec);

	}
	return target;
}


CTOPFeatures* CGUIFeatures::convert_string_word_to_simple_top(CStringFeatures<WORD>* src, CHAR* param)
{
	CTOPFeatures* tf = NULL;

	if (src && src->get_feature_class() == C_SIMPLE && src->get_feature_type() == F_WORD)
	{
		SG_INFO( "converting to TOP features\n");

		if (gui->guihmm.get_pos() && gui->guihmm.get_neg())
		{
			gui->guihmm.get_pos()->set_observations(src);
			gui->guihmm.get_neg()->set_observations(src);

			bool neglinear=false;
			bool poslinear=false;

			tf = new CTOPFeatures(0, gui->guihmm.get_pos(), gui->guihmm.get_neg(), neglinear, poslinear);		     
			ASSERT(tf && tf->set_feature_matrix());
		}
		else
			SG_ERROR( "HMMs not correctly assigned!\n");
	}
	else 
		io.not_implemented();

	return tf;
}

CFKFeatures* CGUIFeatures::convert_string_word_to_simple_fk(CStringFeatures<WORD>* src, CHAR* param)
{
	CFKFeatures* fkf = NULL;

	SG_INFO( "converting to FK features\n");

	if (gui->guihmm.get_pos() && gui->guihmm.get_neg())
	{

		CStringFeatures<WORD>* old_obs_pos=gui->guihmm.get_pos()->get_observations();
		CStringFeatures<WORD>* old_obs_neg=gui->guihmm.get_neg()->get_observations();

		CStringFeatures<WORD>* string_feat = src;
		gui->guihmm.get_pos()->set_observations(string_feat);
		gui->guihmm.get_neg()->set_observations(string_feat);

		fkf = new CFKFeatures(0, gui->guihmm.get_pos(), gui->guihmm.get_neg());//, neglinear, poslinear);		     
		if (train_features)
			fkf->set_opt_a(((CFKFeatures*) train_features)->get_weight_a());
		else
			SG_ERROR( "need train features to set optimal a\n");

		ASSERT(fkf->set_feature_matrix());

		gui->guihmm.get_pos()->set_observations(old_obs_pos);
		gui->guihmm.get_neg()->set_observations(old_obs_neg);
	}
	else
		SG_ERROR( "HMMs not correctly assigned!\n");

	return fkf;
}


CRealFeatures* CGUIFeatures::convert_sparse_real_to_simple_real(CSparseFeatures<DREAL>* src, CHAR* param)
{

	if (src)
	{
		if ( src->get_feature_class() == C_SPARSE)
		{
			if ( src->get_feature_type() == F_DREAL)
			{
				//create dense features with 0 cache
				SG_INFO( "attempting to convert sparse feature matrix to a dense one\n");
				CRealFeatures* rf = new CRealFeatures(0);
				ASSERT(rf);
				INT num_f=0;
				INT num_v=0;
				DREAL* feats=src->get_full_feature_matrix(num_f, num_v);
				rf->set_feature_matrix(feats, num_f, num_v);
				return rf;
			}
		}
		else
			SG_ERROR( "no sparse features available\n");

	}

	SG_ERROR( "conversion failed");
	return NULL;
}

CWordFeatures* CGUIFeatures::convert_simple_char_to_simple_word(CCharFeatures* src, CHAR* param)
{
	CHAR target[1024]="";
	CHAR from_class[1024]="";
	CHAR from_type[1024]="";
	CHAR to_class[1024]="";
	CHAR to_type[1024]="";
	INT order=1;
	INT start=0;
	INT gap = 0 ;
	
	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %s %s %s %s %d %d %d", target, from_class, from_type, to_class, to_type, &order, &start, &gap))<6)
	{
		SG_ERROR( "see help for params (target, from_class, from_type, to_class, to_type, order, start, gap)\n");
		return NULL;
	}

	if ( (src) && (src->get_feature_class() == C_SIMPLE)  && (src->get_feature_type() == F_CHAR) )
	{
		//create dense features with 0 cache
		SG_INFO( "converting CHAR features to WORD ones\n");

		CWordFeatures* wf = new CWordFeatures(0);

		if ( (wf) && (wf->obtain_from_char_features(src, start, order, gap)))
		{
			SG_INFO( "conversion successful\n");
			return wf;
		}

		delete wf;
	}
	else
		SG_ERROR( "no CHAR features available\n");

	SG_ERROR( "conversion failed\n");
	return NULL;
}

CShortFeatures* CGUIFeatures::convert_simple_char_to_simple_short(CCharFeatures* src, CHAR* param)
{
	CHAR target[1024]="";
	CHAR from_class[1024]="";
	CHAR from_type[1024]="";
	CHAR to_class[1024]="";
	CHAR to_type[1024]="";
	INT order=1;
	INT start=0;
	INT gap=0 ;

	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %s %s %s %s %d %d %d", target, from_class, from_type, to_class, to_type, &order, &start, &gap))<6)
		SG_ERROR( "see help for params (target, from_class, from_type, to_class, to_type, order, start, gap)\n");
	
	if (src)
	{
		if ( (src->get_feature_class() == C_SIMPLE)  && (src->get_feature_type() == F_CHAR) )
		{
			//create dense features with 0 cache
			SG_INFO( "converting CHAR features to WORD ones\n");

			CShortFeatures* sf = new CShortFeatures(0);

			if (sf)
			{
				if (sf->obtain_from_char_features(src, start, order, gap))
				{
					SG_INFO( "conversion successful\n");
					return sf;
				}

				delete sf;
			}
		}
		else
			SG_ERROR( "no CHAR features available\n");

	}

	SG_ERROR( "conversion failed\n");
	return NULL;
}



CRealFeatures* CGUIFeatures::convert_simple_char_to_simple_align(CCharFeatures* src, CHAR* param)
{
	CHAR target[1024]="";
	DREAL gapCost=1;
	CHAR from_class[1024]="";
	CHAR from_type[1024]="";
	CHAR to_class[1024]="";
	CHAR to_type[1024]="";

	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %s %s %s %s %le", target, from_class, from_type, to_class, to_type, &gapCost))!=6)
		SG_ERROR( "see help for params\n");

	if ( src &&  (src->get_feature_class() == C_SIMPLE)  && (src->get_feature_type() == F_CHAR) )
	{
		//create dense features with 0 cache
		SG_INFO( "converting CHAR features to REAL ones\n");

		CRealFeatures* rf=new CRealFeatures(0);
		if (rf)
		{
			SG_INFO( "start aligment with gapCost=%1.2f\n", gapCost);
			rf->Align_char_features(src, (CCharFeatures*)ref_features, gapCost);
			SG_INFO( "conversion successful\n");
			return rf;
		}
	}
	else
		SG_ERROR( "no CHAR features available\n");

	SG_ERROR( "conversion failed\n");
	return NULL;
}

bool CGUIFeatures::set_ref_features(CHAR* param)
{
	CHAR target[1024]="";

	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s", target))==1)
	{
		if (strcmp(target,"TRAIN")==0)
		{
			delete ref_features ;
			ref_features = train_features ;
			train_features = NULL ;
			invalidate_train() ;
			return true ;
		}
		else if (strcmp(target,"TEST")==0)
		{
			delete ref_features ;
			ref_features = test_features ;
			test_features = NULL ;
			invalidate_test() ;
			return true ;
		}	  
	}
	SG_ERROR( "see help for params (%s)\n", target);
	return false ;
} ;


bool CGUIFeatures::convert(CHAR* param)
{
	CHAR target[1024]="";
	CHAR from_class[1024]="";
	CHAR from_type[1024]="";
	CHAR to_class[1024]="";
	CHAR to_type[1024]="";
	CFeatures* result = NULL;

	CFeatures** f_ptr=NULL;
	CCombinedFeatures* f_combined=NULL;

	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %s %s %s %s", target, from_class, from_type, to_class, to_type))>=5)
	{
		if (strcmp(target,"TRAIN")==0)
		{
			f_ptr=&train_features;
			invalidate_train() ;
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_features;
			invalidate_test() ;
		}
	}
	else
		SG_ERROR( "see help for params\n");

	if (f_ptr)
	{
		if ( (*f_ptr)->get_feature_class() == C_COMBINED)
		{
			f_combined= (CCombinedFeatures*) (*f_ptr);
			ASSERT(f_combined);

			*f_ptr=f_combined->get_last_feature_obj();
		}

		if (strcmp(from_class, "SIMPLE")==0)
		{
			if (strcmp(from_type, "REAL")==0)
			{
				if (strcmp(to_class, "SPARSE")==0 && strcmp(to_type,"REAL")==0)
					result = convert_simple_real_to_sparse_real(((CRealFeatures*) (*f_ptr)), param);
				else
					io.not_implemented();
			}
			else if (strcmp(from_type, "CHAR")==0)
			{
				if (strcmp(to_class, "STRING")==0 && strcmp(to_type,"CHAR")==0)
					result = convert_simple_char_to_string_char(((CCharFeatures*) (*f_ptr)), param);
				else if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"WORD")==0)
					result = convert_simple_char_to_simple_word(((CCharFeatures*) (*f_ptr)), param);
				else if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"SHORT")==0)
					result = convert_simple_char_to_simple_short(((CCharFeatures*) (*f_ptr)), param);
				else if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"ALIGN")==0)
					result = convert_simple_char_to_simple_align(((CCharFeatures*) (*f_ptr)), param);
				else
					io.not_implemented();
			}
			else if (strcmp(from_type, "WORD")==0)
			{
				if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"SALZBERG")==0)
					result = convert_simple_word_to_simple_salzberg(((CWordFeatures*) (*f_ptr)), param);
				else
					io.not_implemented();
			}
			else
				io.not_implemented();
		}
		else if (strcmp(from_class, "SPARSE")==0)
		{
			if (strcmp(from_type, "REAL")==0)
			{
				if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"REAL")==0)
					result = convert_sparse_real_to_simple_real(((CSparseFeatures<DREAL>*) (*f_ptr)), param);
				else
					io.not_implemented();
			}
			else
				io.not_implemented();
		}
		else if (strcmp(from_class, "STRING")==0)
		{
			if (strcmp(from_type, "CHAR")==0)
			{
				if (strcmp(to_class, "STRING")==0 && strcmp(to_type,"WORD")==0)
					result = convert_string_char_to_string_generic<CHAR,WORD>(((CStringFeatures<CHAR>*) (*f_ptr)), param);
				else if (strcmp(to_class, "STRING")==0 && strcmp(to_type,"ULONG")==0)
					result = convert_string_char_to_string_generic<CHAR,ULONG>(((CStringFeatures<CHAR>*) (*f_ptr)), param);
#ifdef HAVE_MINDY
				else if (strcmp(to_class, "MINDYGRAM")==0 && strcmp(to_type,"ULONG")==0)
					result = convert_string_char_to_mindy_grams<CHAR>(((CStringFeatures<CHAR>*) (*f_ptr)), param);
#endif
				else
					io.not_implemented();
			}
			else if (strcmp(from_type, "BYTE")==0)
			{
				if (strcmp(to_class, "STRING")==0 && strcmp(to_type,"WORD")==0)
					result = convert_string_char_to_string_generic<BYTE,WORD>(((CStringFeatures<BYTE>*) (*f_ptr)), param);
				else if (strcmp(to_class, "STRING")==0 && strcmp(to_type,"ULONG")==0)
					result = convert_string_char_to_string_generic<BYTE,ULONG>(((CStringFeatures<BYTE>*) (*f_ptr)), param);
#ifdef HAVE_MINDY
				else if (strcmp(to_class, "MINDYGRAM")==0 && strcmp(to_type,"ULONG")==0)
					result = convert_string_char_to_mindy_grams<BYTE>(((CStringFeatures<BYTE>*) (*f_ptr)), param);
#endif
				else
					io.not_implemented();
			}
			else if (strcmp(from_type, "WORD")==0)
			{
				if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"TOP")==0)
					result = convert_string_word_to_simple_top(((CStringFeatures<WORD>*) (*f_ptr)), param);
				else 
					io.not_implemented();
			}
			else if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"FK")==0)
				result = convert_string_word_to_simple_fk(((CStringFeatures<WORD>*) (*f_ptr)), param);
			else 
				SG_ERROR( "see help for parameters\n");
		}
		else
			SG_ERROR( "see help for parameters\n");

		if (result)
		{
			SG_INFO( "conversion successful\n");

			delete (*f_ptr);
			(*f_ptr)=result;

			// in case of combined features delete current (==last) feature obj
			// pointer from list (feature object got deleted already above)
			// and append *f_ptr which holds the newly created feature object
			if (f_combined)
			{
				f_combined->delete_feature_obj();
				f_combined->append_feature_obj(*f_ptr);
				*f_ptr=f_combined;
				f_combined->list_feature_objs();
			}
		}
		else
			SG_ERROR( "conversion failed\n");
	}
	else
		SG_ERROR( "no \"%s\" features available\n", target);

	return (result!=NULL);
}

void CGUIFeatures::add_train_features(CFeatures* f)
{
	invalidate_train() ;

	if (!train_features)
	{
		train_features= new CCombinedFeatures();
		ASSERT(train_features);
	}

	if (train_features)
	{
		if (train_features->get_feature_class()!=C_COMBINED)
		{
			CFeatures* first_elem = train_features ;
			train_features= new CCombinedFeatures();
			((CCombinedFeatures*) train_features)->append_feature_obj(first_elem) ;
			((CCombinedFeatures*) train_features)->list_feature_objs();
		}

		ASSERT(f);
		bool result = ((CCombinedFeatures*) train_features)->append_feature_obj(f);
		if (result)
			((CCombinedFeatures*) train_features)->list_feature_objs();
		else
			SG_ERROR( "appending feature object failed\n");
	}
}

void CGUIFeatures::add_test_features(CFeatures* f)
{
	invalidate_test() ;

	if (!test_features)
	{
		test_features= new CCombinedFeatures();
		ASSERT(test_features);
	}

	if (test_features)
	{
		if (test_features->get_feature_class()!=C_COMBINED)
		{
			CFeatures * first_elem = test_features ;
			test_features= new CCombinedFeatures();
			((CCombinedFeatures*)test_features)->append_feature_obj(first_elem) ;
			((CCombinedFeatures*) test_features)->list_feature_objs();	
		}

		ASSERT(f);
		bool result=((CCombinedFeatures*) test_features)->append_feature_obj(f);

		if (result)
			((CCombinedFeatures*) test_features)->list_feature_objs();
		else
			SG_ERROR( "appending feature object failed\n");
	}
	else
		SG_ERROR( "combined feature object could not be created\n");
}
#endif
