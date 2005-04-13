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
			CIO::message(M_ERROR, "see help for parameters\n");
			return false;
		}

		delete (*f_ptr);
		*f_ptr=NULL;

		if (strcmp(fclass, "SIMPLE")==0)
		{
			if (strcmp(type,"REAL")==0)
			{
				*f_ptr=new CRealFeatures(filename);
				//CIO::message("opening file...\n");
				//*f_ptr=new CRealFileFeatures(size, filename);

				//if (comp_features)
				//((CRealFileFeatures*) *f_ptr)->load_feature_matrix() ;
			}
			else if (strcmp(type, "BYTE")==0)
			{
				*f_ptr=new CByteFeatures(filename);
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
				CIO::message(M_ERROR, "unknown type\n");
				return false;
			}
		}
		else if (strcmp(fclass, "SPARSE")==0)
		{
			CIO::not_implemented();
		}
		else if (strcmp(fclass, "STRING")==0)
		{
			if (strcmp(type,"REAL")==0)
			{
				*f_ptr=new CStringFeatures<REAL>(filename);
			}
			else if (strcmp(type, "BYTE")==0)
			{
				*f_ptr=new CStringFeatures<BYTE>(filename);
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
			else
			{
				CIO::message(M_ERROR, "unknown type\n");
				return false;
			}
		}

	} else
		CIO::message(M_ERROR, "see help for params\n");

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
			CIO::message(M_ERROR, "see help for parameters\n");
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
					CIO::message(M_ERROR, "unknown type\n");
					return false;
				}
			}

			if (!result)
				CIO::message(M_ERROR, "writing to file %s failed!\n", fname);
			else
			{
				CIO::message(M_INFO, "successfully written features into \"%s\" !\n", fname);
				result=true;
			}

		} else
			CIO::message(M_ERROR, "set features first\n");

	} else
		CIO::message(M_ERROR, "see help for params\n");

	return result;
}

bool CGUIFeatures::clean(CHAR* param)
{
	param=CIO::skip_spaces(param);
	CHAR target[1024]="";

	if ((sscanf(param, "%s", target))==1)
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
			CIO::message(M_ERROR, "see help for parameters\n");
			return false;
		}

		if (*f_ptr)
		{
			delete *f_ptr ;
			*f_ptr = NULL ;
			return true ;
		} else
			CIO::message(M_DEBUG, "feature already = NULL\n") ;
		return false ;
	} else
		CIO::message(M_ERROR, "see help for params\n");

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
		CIO::message(M_ERROR, "see help for params\n");

	if (f_ptr)
	{
		CIO::message(M_INFO, "reshape data to %d x %d\n", num_feat, num_vec);
		result=(*f_ptr)->reshape(num_feat, num_vec);

		if (!result)
			CIO::message(M_ERROR, "reshaping failed");
	}

	return result;
}

CSparseRealFeatures* CGUIFeatures::convert_simple_real_to_sparse_real(CRealFeatures* src, CHAR* param)
{
	if (src)
	{
		if ( (src->get_feature_class()) == C_SIMPLE)
		{
			if ( (src->get_feature_type()) == F_REAL)
			{
				//create sparse features with 0 cache
				CIO::message(M_INFO, "attempting to convert dense feature matrix to a sparse one\n");
				CSparseRealFeatures* target=new CSparseRealFeatures(0l);
				INT num_f=0;
				INT num_v=0;
				REAL* feats=src->get_feature_matrix(num_f, num_v);
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


	int num_symbols = 4; //DNA is default

	switch (src->get_alphabet())
	{
		case DNA:
			num_symbols = 4;
			break;
		case PROTEIN:
			num_symbols = 26;
			break;
		case ALPHANUM:
			num_symbols = 36;
			break;
		case CUBE:
			num_symbols = 6;
			break;
		case NONE:
			num_symbols = 0;
			break;
		default:
			num_symbols = 4;
	}

	CStringFeatures<CHAR>* target= new CStringFeatures<CHAR>(num_symbols);
	target->set_features(strings, num_vec, max_len, num_symbols, src->get_alphabet());

	return target;
}

CRealFeatures* CGUIFeatures::convert_simple_word_to_simple_salzberg(CWordFeatures* src, CHAR* param)
{
	CPluginEstimate* pie=gui->guipluginestimate.get_estimator();
	assert(src->get_feature_type()==F_WORD && src->get_feature_class()==C_SIMPLE);
	assert(pie);

	CRealFeatures* target=new CRealFeatures(0l);
	assert(target);

	INT num_feat=src->get_num_features();
	INT num_vec=src->get_num_vectors();
	REAL* fm=new REAL[num_vec*num_feat];

	if (fm)
	{
		for (INT i=0; i<num_vec; i++)
		{
			INT len=0;
			bool to_free=false;
			WORD* vec = src->get_feature_vector(i, len, to_free);
			assert(num_feat==len);

			for (INT j=0; j<num_feat; j++)
				fm[i*num_feat+j]=pie->get_parameterwise_log_odds(vec[j], j);

			src->free_feature_vector(vec, i, to_free);
		}
		target->set_feature_matrix(fm, num_feat, num_vec);

	}
	return target;
}

CStringFeatures<ULONG>* CGUIFeatures::convert_string_char_to_string_ulong(CStringFeatures<CHAR>* src, CHAR* param)
{
	CHAR which[1024]="";
	CHAR from_class[1024]="";
	CHAR from_type[1024]="";
	CHAR to_class[1024]="";
	CHAR to_type[1024]="";
	INT order=1;
	INT start=0;

	if ((sscanf(param, "%s %s %s %s %s %d %d", which, from_class, from_type, to_class, to_type, &order, &start))>=5)
	{
		if ( ( (src->get_feature_class()) == C_STRING)  && ( (src->get_feature_type()) == F_CHAR) )
		{
			//create dense features with 0 cache
			CIO::message(M_INFO, "converting CHAR STRING features to ULONG STRING ones (order=%i)\n",order);

			CStringFeatures<ULONG>* sf=new CStringFeatures<ULONG>();
			if (sf && sf->obtain_from_char_features(src, start, order))
				return sf;

			delete sf;
		}
		else
			CIO::message(M_ERROR, "features are not of class/type STRING/CHAR\n");
	}
	else
		CIO::message(M_ERROR, "see help for parameters\n");

	return NULL;
}

CStringFeatures<WORD>* CGUIFeatures::convert_string_char_to_string_word(CStringFeatures<CHAR>* src, CHAR* param)
{
	CHAR which[1024]="";
	CHAR from_class[1024]="";
	CHAR from_type[1024]="";
	CHAR to_class[1024]="";
	CHAR to_type[1024]="";
	INT order=1;
	INT start=0;

	if ((sscanf(param, "%s %s %s %s %s %d %d", which, from_class, from_type, to_class, to_type, &order, &start))>=5)
	{
		if ( ( (src->get_feature_class()) == C_STRING)  && ( (src->get_feature_type()) == F_CHAR) )
		{
			//create dense features with 0 cache
			CIO::message(M_INFO, "converting CHAR STRING features to WORD STRING ones (order=%i)\n",order);

			CStringFeatures<WORD>* sf=new CStringFeatures<WORD>();
			if (sf && sf->obtain_from_char_features(src, start, order))
				return sf;

			delete sf;
		}
		else
			CIO::message(M_ERROR, "features are not of class/type STRING/CHAR\n");
	}
	else
		CIO::message(M_ERROR, "see help for parameters\n");

	return NULL;
}

CTOPFeatures* CGUIFeatures::convert_string_word_to_simple_top(CStringFeatures<WORD>* src, CHAR* param)
{
	CTOPFeatures* tf = NULL;

	if (src && src->get_feature_class() == C_SIMPLE && src->get_feature_type() == F_WORD)
	{
		CIO::message(M_INFO, "converting to TOP features\n");

		if (gui->guihmm.get_pos() && gui->guihmm.get_neg())
		{
			gui->guihmm.get_pos()->set_observations(src);
			gui->guihmm.get_neg()->set_observations(src);

			bool neglinear=false;
			bool poslinear=false;

			tf = new CTOPFeatures(0l, gui->guihmm.get_pos(), gui->guihmm.get_neg(), neglinear, poslinear);		     
			assert(tf && tf->set_feature_matrix());
		}
		else
			CIO::message(M_ERROR, "HMMs not correctly assigned!\n");
	}
	else 
		CIO::not_implemented();

	return tf;
}

CFKFeatures* CGUIFeatures::convert_string_word_to_simple_fk(CStringFeatures<WORD>* src, CHAR* param)
{
	CFKFeatures* fkf = NULL;

	CIO::message(M_INFO, "converting to FK features\n");

	if (gui->guihmm.get_pos() && gui->guihmm.get_neg())
	{

		CStringFeatures<WORD>* old_obs_pos=gui->guihmm.get_pos()->get_observations();
		CStringFeatures<WORD>* old_obs_neg=gui->guihmm.get_neg()->get_observations();

		CStringFeatures<WORD>* string_feat = src;
		gui->guihmm.get_pos()->set_observations(string_feat);
		gui->guihmm.get_neg()->set_observations(string_feat);

		CFKFeatures* fkf = new CFKFeatures(0l, gui->guihmm.get_pos(), gui->guihmm.get_neg());//, neglinear, poslinear);		     
		if (train_features)
			fkf->set_opt_a(((CFKFeatures*) train_features)->get_weight_a());
		else
			CIO::message(M_ERROR, "need train features to set optimal a\n");

		assert(fkf->set_feature_matrix());

		gui->guihmm.get_pos()->set_observations(old_obs_pos);
		gui->guihmm.get_neg()->set_observations(old_obs_neg);
	}
	else
		CIO::message(M_ERROR, "HMMs not correctly assigned!\n");

	return fkf;
}


CRealFeatures* CGUIFeatures::convert_sparse_real_to_simple_real(CSparseRealFeatures* src, CHAR* param)
{

	if (src)
	{
		if ( src->get_feature_class() == C_SPARSE)
		{
			if ( src->get_feature_type() == F_REAL)
			{
				//create dense features with 0 cache
				CIO::message(M_INFO, "attempting to convert sparse feature matrix to a dense one\n");
				CRealFeatures* rf = new CRealFeatures(0l);
				assert(rf);
				INT num_f=0;
				INT num_v=0;
				REAL* feats=src->get_full_feature_matrix(num_f, num_v);
				rf->set_feature_matrix(feats, num_f, num_v);
				return rf;
			}
		}
		else
			CIO::message(M_ERROR, "no sparse features available\n");

	}

	CIO::message(M_ERROR, "conversion failed");
	return NULL;
}

CWordFeatures* CGUIFeatures::convert_simple_char_to_simple_word(CCharFeatures* src, CHAR* param)
{
	CHAR target[1024]="";
	CHAR from_class[1024]="";
	CHAR from_type[1024]="";
	CHAR to_class[1024]="";
	CHAR to_type[1024]="";
	CHAR alpha[1024]="";
	INT order=1;
	INT start=0;
	E_ALPHABET alphabet=DNA;


	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %s %s %s %s %s %d %d", target, from_class, from_type, to_class, to_type, alpha, &order, &start))>=6)
	{
		if (strcmp(alpha,"PROTEIN")==0)
			alphabet=PROTEIN;
		else if (strcmp(alpha,"ALPHANUM")==0)
			alphabet=ALPHANUM;
		else if (strcmp(alpha,"DNA")==0)
			alphabet=DNA;
		else if (strcmp(alpha,"CUBE")==0)
			alphabet=CUBE;
		else
			CIO::message(M_ERROR, "unknown alphabet!\n");
	}
	else
		CIO::message(M_ERROR, "see help for params\n");

	if (src)
	{
		if ( (src->get_feature_class() == C_SIMPLE)  && (src->get_feature_type() == F_CHAR) )
		{
			//create dense features with 0 cache
			CIO::message(M_INFO, "converting CHAR features to WORD ones\n");

			CWordFeatures* wf = new CWordFeatures(0l);

			if (wf)
			{
				if (wf->obtain_from_char_features(src, alphabet, start, order))
				{
					CIO::message(M_INFO, "conversion successful\n");
					return wf;
				}

				delete wf;
			}
		}
		else
			CIO::message(M_ERROR, "no CHAR features available\n");

	}

	CIO::message(M_ERROR, "conversion failed\n");
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

	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %s %s %s %s %d %d", target, from_class, from_type, to_class, to_type, &order, &start))<5)
		CIO::message(M_ERROR, "see help for params\n");

	if (src)
	{
		if ( (src->get_feature_class() == C_SIMPLE)  && (src->get_feature_type() == F_CHAR) )
		{
			//create dense features with 0 cache
			CIO::message(M_INFO, "converting CHAR features to WORD ones\n");

			CShortFeatures* sf = new CShortFeatures(0l);

			if (sf)
			{
				if (sf->obtain_from_char_features(src, start, order))
				{
					CIO::message(M_INFO, "conversion successful\n");
					return sf;
				}

				delete sf;
			}
		}
		else
			CIO::message(M_ERROR, "no CHAR features available\n");

	}

	CIO::message(M_ERROR, "conversion failed\n");
	return NULL;
}



CRealFeatures* CGUIFeatures::convert_simple_char_to_simple_align(CCharFeatures* src, CHAR* param)
{
	CHAR target[1024]="";
	REAL gapCost=1;
	CHAR from_class[1024]="";
	CHAR from_type[1024]="";
	CHAR to_class[1024]="";
	CHAR to_type[1024]="";

	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %s %s %s %s %le", target, from_class, from_type, to_class, to_type, &gapCost))!=6)
		CIO::message(M_ERROR, "see help for params\n");

	if ( src &&  (src->get_feature_class() == C_SIMPLE)  && (src->get_feature_type() == F_CHAR) )
	{
		//create dense features with 0 cache
		CIO::message(M_INFO, "converting CHAR features to REAL ones\n");

		CRealFeatures* rf=new CRealFeatures(0l);
		if (rf)
		{
			CIO::message(M_INFO, "start aligment with gapCost=%1.2f\n", gapCost);
			rf->Align_char_features(src, (CCharFeatures*)ref_features, gapCost);
			CIO::message(M_INFO, "conversion successful\n");
			return rf;
		}
	}
	else
		CIO::message(M_ERROR, "no CHAR features available\n");

	CIO::message(M_ERROR, "conversion failed\n");
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
	CIO::message(M_ERROR, "see help for params (%s)\n", target);
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
		CIO::message(M_ERROR, "see help for params\n");

	if (f_ptr)
	{
		if ( (*f_ptr)->get_feature_class() == C_COMBINED)
		{
			f_combined= (CCombinedFeatures*) (*f_ptr);
			assert(f_combined);

			*f_ptr=f_combined->get_last_feature_obj();
		}

		if (strcmp(from_class, "SIMPLE")==0)
		{
			if (strcmp(from_type, "REAL")==0)
			{
				if (strcmp(to_class, "SPARSE")==0 && strcmp(to_type,"REAL")==0)
					result = convert_simple_real_to_sparse_real(((CRealFeatures*) (*f_ptr)), param);
				else
					CIO::not_implemented();
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
					CIO::not_implemented();
			}
			else if (strcmp(from_type, "WORD")==0)
			{
				if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"SALZBERG")==0)
					result = convert_simple_word_to_simple_salzberg(((CWordFeatures*) (*f_ptr)), param);
				else
					CIO::not_implemented();
			}
			else
				CIO::not_implemented();
		}
		else if (strcmp(from_class, "SPARSE")==0)
		{
			if (strcmp(from_type, "REAL")==0)
			{
				if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"REAL")==0)
					result = convert_sparse_real_to_simple_real(((CSparseRealFeatures*) (*f_ptr)), param);
				else
					CIO::not_implemented();
			}
			else
				CIO::not_implemented();
		}
		else if (strcmp(from_class, "STRING")==0)
		{
			if (strcmp(from_type, "CHAR")==0)
			{
				if (strcmp(to_class, "STRING")==0 && strcmp(to_type,"WORD")==0)
					result = convert_string_char_to_string_word(((CStringFeatures<CHAR>*) (*f_ptr)), param);
				else if (strcmp(to_class, "STRING")==0 && strcmp(to_type,"ULONG")==0)
					result = convert_string_char_to_string_ulong(((CStringFeatures<CHAR>*) (*f_ptr)), param);
				else
					CIO::not_implemented();
			}
			else if (strcmp(from_type, "WORD")==0)
			{
				if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"TOP")==0)
					result = convert_string_word_to_simple_top(((CStringFeatures<WORD>*) (*f_ptr)), param);
				else 
					CIO::not_implemented();
			}
			else if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"FK")==0)
				result = convert_string_word_to_simple_fk(((CStringFeatures<WORD>*) (*f_ptr)), param);
			else 
				CIO::message(M_ERROR, "see help for parameters\n");
		}
		else
			CIO::message(M_ERROR, "see help for parameters\n");

		if (result)
		{
			CIO::message(M_INFO, "conversion successful\n");

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
			CIO::message(M_ERROR, "conversion failed\n");
	}
	else
		CIO::message(M_ERROR, "no \"%s\" features available\n", target);

	return (result!=NULL);
}

void CGUIFeatures::add_train_features(CFeatures* f)
{
	invalidate_train() ;

	if (!train_features)
	{
		train_features= new CCombinedFeatures();
		assert(train_features);
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

		assert(f);
		bool result = ((CCombinedFeatures*) train_features)->append_feature_obj(f);
		assert(result) ;
		((CCombinedFeatures*) train_features)->list_feature_objs();
	}
}

void CGUIFeatures::add_test_features(CFeatures* f)
{
	invalidate_test() ;

	if (!test_features)
	{
		test_features= new CCombinedFeatures();
		assert(test_features);
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

		assert(f);
		bool result=((CCombinedFeatures*) test_features)->append_feature_obj(f);
		assert(result);
		((CCombinedFeatures*) test_features)->list_feature_objs();
	}
	else
		CIO::message(M_ERROR, "combined feature object could not be created\n");
}
