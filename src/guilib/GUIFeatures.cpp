#include "guilib/GUIFeatures.h"
#include "gui/GUI.h"
#include "lib/io.h"
#include "features/RealFileFeatures.h"
#include "features/TOPFeatures.h"
#include "features/FKFeatures.h"
#include "features/CharFeatures.h"
#include "features/StringFeatures.h"
#include "features/ByteFeatures.h"
#include "features/ShortFeatures.h"
#include "features/RealFeatures.h"
#include "features/SparseRealFeatures.h"
#include "features/CombinedFeatures.h"
#include "features/Features.h"

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
		
bool CGUIFeatures::preprocess(CHAR* param)
{
	bool result=false;
	param=CIO::skip_spaces(param);
	CHAR target[1024]="";
	INT force=0;

	if ((sscanf(param, "%s %d", target, &force))>=1)
	{
		if ( strcmp(target, "TRAIN")==0 || strcmp(target, "TEST")==0 )
		{
			if (strcmp(target,"TRAIN")==0)
			{
				preprocess_features(train_features, NULL, force==1);
			}
			else if (strcmp(target,"TEST")==0)
			{
				preprocess_features(train_features, test_features, force==1);
			}
			else
				CIO::message(M_ERROR, "see help for parameters\n");
		}
		else
			CIO::message(M_ERROR, "features not correctly assigned!\n");
	}
	else
		CIO::message(M_ERROR, "see help for parameters\n");

	return result;
}

bool CGUIFeatures::preprocess_features(CFeatures* trainfeat, CFeatures* testfeat, bool force)
{
	INT num_preproc=0;
	CPreProc** preprocs;
	if ((preprocs=gui->guipreproc.get_preprocs(num_preproc))!=NULL)
	{
		if (trainfeat)
		{
			if (testfeat)
			{
				assert(trainfeat->get_num_preproc()==num_preproc);

				for (INT i=0; i<trainfeat->get_num_preproc();  i++)
				{
					preprocs[i]->init(trainfeat);
					testfeat->add_preproc(trainfeat->get_preproc(i));
				}

				preproc_all_features(testfeat, force);
			}
			else
			{
				for (INT i=0; i<num_preproc; i++)
				{
					preprocs[i]->init(trainfeat);
					trainfeat->add_preproc(preprocs[i]);

					preproc_all_features(trainfeat, force);
				}
			}

			return true;
		}
		else
			CIO::message(M_ERROR, "no features for preprocessing available!\n");
	}
	else
		CIO::message(M_ERROR, "no preprocessors available!\n");

	return false;
}

bool CGUIFeatures::preproc_all_features(CFeatures* f, bool force)
{
	switch (f->get_feature_class())
	{
		case C_SIMPLE:
			switch (f->get_feature_type())
			{
				case F_REAL:
					return ((CRealFeatures*) f)->preproc_feature_matrix(force);
				case F_SHORT:
					return ((CShortFeatures*) f)->preproc_feature_matrix(force);
				case F_WORD:
					return ((CShortFeatures*) f)->preproc_feature_matrix(force);
				case F_CHAR:
					return ((CCharFeatures*) f)->preproc_feature_matrix(force);
				case F_BYTE:
					return ((CByteFeatures*) f)->preproc_feature_matrix(force);
				default:
					CIO::not_implemented();
			}
			break;
		case C_SPARSE:
			switch (f->get_feature_type())
			{
				case F_REAL:
					return ((CSparseRealFeatures*) f)->preproc_feature_matrix(force);
				default:
					CIO::not_implemented();
			};
			break;
		default:
			CIO::not_implemented();
	}

	return false;
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
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_features;
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

bool CGUIFeatures::convert_full_to_sparse(CHAR* param)
{
	bool result=false;
	CHAR target[1024]="";

	CFeatures** f_ptr=NULL;

	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s", target))==1)
	{
		if (strcmp(target,"TRAIN")==0)
		{
			f_ptr=&train_features;
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_features;
		}
	}
	else
		CIO::message(M_ERROR, "see help for params\n");

	if (f_ptr)
	{
		if ( ((*f_ptr)->get_feature_class()) == C_SIMPLE)
		{
			if ( ((*f_ptr)->get_feature_type()) == F_REAL)
			{
				//create sparse features with 0 cache
				CIO::message(M_INFO, "attempting to convert dense feature matrix to a sparse one\n");
				CSparseRealFeatures* sf=new CSparseRealFeatures(0l);
				INT num_f=0;
				INT num_v=0;
				REAL* feats=((CRealFeatures*)(*f_ptr))->get_feature_matrix(num_f, num_v);
				result=sf->set_full_feature_matrix(feats, num_f, num_v);
				delete (*f_ptr);
				(*f_ptr)=sf;
			}
		}
		else
			CIO::message(M_ERROR, "no Simple features available\n");

		if (!result)
			CIO::message(M_ERROR, "conversion failed");
	}

	return result;
}

bool CGUIFeatures::convert_sparse_to_full(CHAR* param)
{
	bool result=false;
	CHAR target[1024]="";

	CFeatures** f_ptr=NULL;

	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s", target))==1)
	{
		if (strcmp(target,"TRAIN")==0)
		{
			f_ptr=&train_features;
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_features;
		}
	}
	else
		CIO::message(M_ERROR, "see help for params\n");

	if (f_ptr)
	{
		if ( ((*f_ptr)->get_feature_class()) == C_SPARSE)
		{
			if ( ((*f_ptr)->get_feature_type()) == F_REAL)
			{
				//create dense features with 0 cache
				CIO::message(M_INFO, "attempting to convert sparse feature matrix to a dense one\n");
				CRealFeatures* sf=new CRealFeatures(0l);
				INT num_f=0;
				INT num_v=0;
				REAL* feats=((CSparseRealFeatures*)(*f_ptr))->get_full_feature_matrix(num_f, num_v);
				result=(feats!=NULL);
				sf->set_feature_matrix(feats, num_f, num_v);
				delete (*f_ptr);
				(*f_ptr)=sf;
			}
		}
		else
			CIO::message(M_ERROR, "no sparse features available\n");

		if (!result)
			CIO::message(M_ERROR, "conversion failed");
	}

	return result;
}

bool CGUIFeatures::convert_char_to_word(CHAR* param)
{
	bool result=false;
	CHAR target[1024]="";
	CHAR alpha[1024]="";
	INT order=1;
	INT start=0;
	E_ALPHABET alphabet=DNA;

	CFeatures** f_ptr=NULL;
	
	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %s %d %d", target, alpha, &order, &start))>=3)
	{
		if (strcmp(target,"TRAIN")==0)
		{
			f_ptr=&train_features;
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_features;
		}

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

	if (f_ptr)
	{
		if ( (*f_ptr) && ( ((*f_ptr)->get_feature_class()) == C_SIMPLE)  && ( ((*f_ptr)->get_feature_type()) == F_CHAR) )
		{
			//create dense features with 0 cache
			CIO::message(M_INFO, "converting CHAR features to WORD ones\n");

			CWordFeatures* wf=new CWordFeatures(0l);
			result=(wf!=NULL);

			if (result)
			{
				wf->obtain_from_char_features((CCharFeatures*) (*f_ptr), alphabet, start, order);
				delete (*f_ptr);
				(*f_ptr)=wf;
			}
		}
		else
			CIO::message(M_ERROR, "no CHAR features available\n");

		if (!result)
			CIO::message(M_ERROR, "conversion failed\n");
		else
			CIO::message(M_INFO, "conversion successful\n");
	}

	return result;
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
	      return true ;
	    }
	  else if (strcmp(target,"TEST")==0)
	    {
	      delete ref_features ;
	      ref_features = test_features ;
	      test_features = NULL ;
	      return true ;
	    }	  
	}
	CIO::message(M_ERROR, "see help for params (%s)\n", target);
	return false ;
} ;


bool CGUIFeatures::alignment_char(CHAR* param)
{
	bool result=false;
	CHAR target[1024]="";
	REAL gapCost=1;

	CFeatures** f_ptr=NULL;
	
	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %le", target, &gapCost))==2)
	  {
	    if (strcmp(target,"TRAIN")==0)
	      {
		f_ptr=&train_features;
	      }
	    else if (strcmp(target,"TEST")==0)
	      {
		f_ptr=&test_features;
	      }	  
	  }
	else
	  CIO::message(M_ERROR, "see help for params %s\n", target);

	if (f_ptr)
	  {
	    if ( (*f_ptr) && ( ((*f_ptr)->get_feature_class()) == C_SIMPLE)  && ( ((*f_ptr)->get_feature_type()) == F_CHAR) )
	      {
		//create dense features with 0 cache
		CIO::message(M_INFO, "converting CHAR features to REAL ones\n");
		
		CRealFeatures* rf=new CRealFeatures(0l);
		result=(rf!=NULL);
		
		if (result)
		  {
		    CIO::message(M_INFO, "start aligment with gapCost=%1.2f\n", gapCost);
		    rf->Align_char_features((CCharFeatures*) (*f_ptr), (CCharFeatures*)ref_features, gapCost);
		    delete (*f_ptr);
		    (*f_ptr)=rf;
		  }
	      }
	    else
	      CIO::message(M_ERROR, "no CHAR features available\n");
	    
	    if (!result)
	      CIO::message(M_ERROR, "conversion failed\n");
	    else
	      CIO::message(M_INFO, "conversion successful\n");
	  }

	return result;
}

bool CGUIFeatures::convert_char_to_short(CHAR* param)
{
	bool result=false;
	CHAR target[1024]="";
	CHAR alpha[1024]="";
	INT order=1;
	INT start=0;
	E_ALPHABET alphabet=DNA;

	CFeatures** f_ptr=NULL;
	
	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %s %d %d", target, alpha, &order, &start))>=3)
	{
		if (strcmp(target,"TRAIN")==0)
		{
			f_ptr=&train_features;
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_features;
		}

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

	if (f_ptr)
	{
		if ( ( ((*f_ptr)->get_feature_class()) == C_SIMPLE)  && ( ((*f_ptr)->get_feature_type()) == F_CHAR) )
		{
			//create dense features with 0 cache
			CIO::message(M_INFO, "converting CHAR features to SHORT ones\n");

			CShortFeatures* sf=new CShortFeatures(0l);
			result=(sf!=NULL);

			if (result)
			{
				sf->obtain_from_char_features((CCharFeatures*) (*f_ptr), alphabet, start, order);
				delete (*f_ptr);
				(*f_ptr)=sf;
			}
		}
		else
			CIO::message(M_ERROR, "no CHAR features available\n");

		if (!result)
			CIO::message(M_ERROR, "conversion failed\n");
		else
			CIO::message(M_ERROR, "conversion successful\n");
	}

	return result;
}

bool CGUIFeatures::convert(CHAR* param)
{
	bool result=false;
	CHAR target[1024]="";
	CHAR from_class[1024]="";
	CHAR from_type[1024]="";
	CHAR to_class[1024]="";
	CHAR to_type[1024]="";

	CFeatures** f_ptr=NULL;

	param=CIO::skip_spaces(param);
	if ((sscanf(param, "%s %s %s %s %s", target, from_class, from_type, to_class, to_type))>=5)
	{
		if (strcmp(target,"TRAIN")==0)
		{
			f_ptr=&train_features;
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_features;
		}
	}
	else
		CIO::message(M_ERROR, "see help for params\n");

	if (f_ptr)
	{
		if (strcmp(from_class, "SIMPLE")==0)
		{
			if (strcmp(from_type, "CHAR")==0)
			{
				if (strcmp(to_class, "STRING")==0 && strcmp(to_type,"CHAR")==0)
				{
					INT num_vec=((CCharFeatures*) (*f_ptr))->get_num_vectors();
					T_STRING<CHAR>* strings=new T_STRING<CHAR>[num_vec];
					INT max_len=-1;

					for (INT i=0; i<num_vec; i++)
					{
						bool to_free=false;
						INT len=0;
						CHAR* str=((CCharFeatures*) (*f_ptr))->get_feature_vector(i, len, to_free);
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

						((CCharFeatures*) (*f_ptr))->free_feature_vector(str, i, to_free);
					}

					/// FIXME the 4
					CStringFeatures<CHAR>* string_feat= new CStringFeatures<CHAR>(4);
					string_feat->set_features(strings, num_vec, max_len, 4, ((CCharFeatures*) (*f_ptr))->get_alphabet());

					delete (*f_ptr);
					(*f_ptr)=string_feat;
					result= string_feat!=NULL;
				}
				else
					CIO::not_implemented();

			}
			else if (strcmp(from_type, "WORD")==0)
			{
				if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"SALZBERG")==0)
				{
					CPluginEstimate* pie=gui->guipluginestimate.get_estimator();
					assert((*f_ptr)->get_feature_type()==F_WORD && (*f_ptr)->get_feature_class()==C_SIMPLE);
					assert(pie);

					CRealFeatures* real_feat=new CRealFeatures(0l);
					assert(real_feat);

					INT num_feat=((CWordFeatures*) (*f_ptr))->get_num_features();
					INT num_vec=((CWordFeatures*) (*f_ptr))->get_num_vectors();
					REAL* fm=new REAL[num_vec*num_feat];

					if (fm)
					{
						for (INT i=0; i<num_vec; i++)
						{
							INT len=0;
							bool to_free=false;
							WORD* vec=((CWordFeatures*) (*f_ptr))->get_feature_vector(i, len, to_free);
							assert(num_feat==len);
							for (INT j=0; j<num_feat; j++)
							{
								fm[i*num_feat+j]=pie->get_parameterwise_log_odds(vec[j], j);
							}

							((CWordFeatures*) (*f_ptr))->free_feature_vector(vec, i, to_free);
						}
						real_feat->set_feature_matrix(fm, num_feat, num_vec);

						delete (*f_ptr);
						(*f_ptr)=real_feat;
					}
					result= real_feat!=NULL;
				}
				else
					CIO::not_implemented();
			}
			else
				CIO::not_implemented();
		}
		else if (strcmp(from_class, "SPARSE")==0)
		{
			CIO::not_implemented();
		}
		else if (strcmp(from_class, "STRING")==0)
		{
			if (strcmp(from_type, "CHAR")==0)
			{
				if (strcmp(to_class, "STRING")==0 && strcmp(to_type,"WORD")==0)
				{
					CHAR alpha[1024]="";
					INT order=1;
					INT start=0;
					E_ALPHABET alphabet=DNA;
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

						if ( ( ((*f_ptr)->get_feature_class()) == C_STRING)  && ( ((*f_ptr)->get_feature_type()) == F_CHAR) )
						{
							//create dense features with 0 cache
							CIO::message(M_ERROR, "converting CHAR STRING features to WORD STRING ones\n");

							CStringFeatures<WORD>* sf=new CStringFeatures<WORD>();
							result=(sf!=NULL);

							CCharFeatures cf(alphabet, 0l);

							if (result)
							{
								sf->obtain_from_char_features((CStringFeatures<CHAR>*) (*f_ptr), &cf, alphabet, start, order);
								delete (*f_ptr);
								(*f_ptr)=sf;
							}
						}
						else
							CIO::message(M_ERROR, "features are not of class/type STRING/CHAR\n");
					}
					else
						CIO::message(M_ERROR, "see help for parameters\n");
				}
				else
					CIO::not_implemented();
			}
			else if (strcmp(from_type, "WORD")==0)
			{
				if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"TOP")==0)
				{
					CIO::message(M_INFO, "converting to TOP features\n");
					if (gui->guihmm.get_pos() && gui->guihmm.get_neg())
					{

						//CStringFeatures<WORD>* old_obs_pos=gui->guihmm.get_pos()->get_observations();
						//CStringFeatures<WORD>* old_obs_neg=gui->guihmm.get_neg()->get_observations();

						CStringFeatures<WORD>* string_feat=(CStringFeatures<WORD>*) (*f_ptr);
						gui->guihmm.get_pos()->set_observations(string_feat);
						gui->guihmm.get_neg()->set_observations(string_feat);

						bool neglinear=false;
						bool poslinear=false;

						CTOPFeatures* top_feat= new CTOPFeatures(0l, gui->guihmm.get_pos(), gui->guihmm.get_neg(), neglinear, poslinear);		     
						assert(top_feat->set_feature_matrix());

						//gui->guihmm.get_pos()->set_observations(old_obs_pos);
						//gui->guihmm.get_neg()->set_observations(old_obs_neg);

						delete (*f_ptr);
						*f_ptr=top_feat;
						result=true;
					}
					else
						CIO::message(M_ERROR, "HMMs not correctly assigned!\n");
				}
				else 
					CIO::not_implemented();
			}
			else if (strcmp(to_class, "SIMPLE")==0 && strcmp(to_type,"FK")==0)
			{
				CIO::message(M_INFO, "converting to FK features\n");
				if (gui->guihmm.get_pos() && gui->guihmm.get_neg())
				{

					CStringFeatures<WORD>* old_obs_pos=gui->guihmm.get_pos()->get_observations();
					CStringFeatures<WORD>* old_obs_neg=gui->guihmm.get_neg()->get_observations();

					CStringFeatures<WORD>* string_feat=(CStringFeatures<WORD>*) (*f_ptr);
					gui->guihmm.get_pos()->set_observations(string_feat);
					gui->guihmm.get_neg()->set_observations(string_feat);

					//bool neglinear=false;
					//bool poslinear=false;

					CFKFeatures* top_feat= new CFKFeatures(0l, gui->guihmm.get_pos(), gui->guihmm.get_neg());//, neglinear, poslinear);		     
					if (train_features)
						((CFKFeatures*) *f_ptr)->set_opt_a(((CFKFeatures*) train_features)->get_weight_a());
					else
						CIO::message(M_ERROR, "need train features to set optimal a\n");

					assert(top_feat->set_feature_matrix());

					gui->guihmm.get_pos()->set_observations(old_obs_pos);
					gui->guihmm.get_neg()->set_observations(old_obs_neg);

					delete (*f_ptr);
					*f_ptr=top_feat;
				}
				else
					CIO::message(M_ERROR, "HMMs not correctly assigned!\n");
			}
			else 
				CIO::message(M_ERROR, "see help for parameters\n");
		}
		else
			CIO::message(M_ERROR, "see help for parameters\n");

		if (!result)
			CIO::message(M_ERROR, "conversion failed\n");
		else
			CIO::message(M_INFO, "conversion successful\n");

	}
	else
		CIO::message(M_ERROR, "no \"%s\" features available\n", target);

	return result;
}

void CGUIFeatures::add_train_features(CFeatures* f)
{
	if (!train_features || (train_features && train_features->get_feature_class()!=C_COMBINED))
	{
		delete train_features;
		train_features= new CCombinedFeatures();
		assert(train_features);
	}

	if (train_features)
	{
		assert(f);
		assert(((CCombinedFeatures*) train_features)->append_feature_obj(f));
		((CCombinedFeatures*) train_features)->list_feature_objs();
	}
	else
		CIO::message(M_ERROR, "combined feature object could not be created\n");
}

void CGUIFeatures::add_test_features(CFeatures* f)
{
	if (!test_features || (test_features && test_features->get_feature_class()!=C_COMBINED))
	{
		delete test_features;
		test_features= new CCombinedFeatures();
		assert(test_features);
	}

	if (test_features)
	{
		assert(f);
		assert(((CCombinedFeatures*) test_features)->append_feature_obj(f));
		((CCombinedFeatures*) test_features)->list_feature_objs();
	}
	else
		CIO::message(M_ERROR, "combined feature object could not be created\n");
}
