#include "guilib/GUIPreProc.h"
#include "gui/GUI.h"
#include "preproc/LogPlusOne.h"
#include "preproc/NormOne.h"
#include "preproc/PruneVarSubMean.h"
#include "preproc/PCACut.h"
#include "preproc/SortWord.h"
#include "preproc/SortWordString.h"
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
#include "lib/io.h"
#include "lib/config.h"

#include <string.h>
#include <stdio.h>
#include <assert.h>

CGUIPreProc::CGUIPreProc(CGUI * gui_)
  : gui(gui_)
{
	preprocs=new CList<CPreProc*>(true);
	attached_preprocs_lists=new CList<CList<CPreProc*>*>(true);
}

CGUIPreProc::~CGUIPreProc()
{
	delete preprocs;
	delete attached_preprocs_lists;
}


bool CGUIPreProc::add_preproc(CHAR* param)
{
	CPreProc* preproc=NULL;

	param=CIO::skip_spaces(param);
#ifdef HAVE_ATLAS
#ifdef HAVE_LAPACK
	if (strncmp(param,"PCACUT",6)==0)
	{
		INT do_whitening=0; 
		double thresh=1e-6 ;
		sscanf(param+6, "%i %le", &do_whitening, &thresh) ;
		CIO::message(M_INFO, "PCACUT parameters: do_whitening=%i thresh=%e", do_whitening, thresh) ;
		preproc=new CPCACut(do_whitening, thresh);
	}
	else 
#endif // LAPACK
#endif // ATLAS
	if (strncmp(param,"NORMONE",7)==0)
	{
		preproc=new CNormOne();
	}
	else if (strncmp(param,"LOGPLUSONE",10)==0)
	{
		preproc=new CLogPlusOne();
	}
	else if (strncmp(param,"SORTWORDSTRING",14)==0)
	{
		preproc=new CSortWordString();
	}
	else if (strncmp(param,"SORTWORD",8)==0)
	{
		preproc=new CSortWord();
	}
	else if (strncmp(param,"PRUNEVARSUBMEAN",15)==0)
	{
		INT divide_by_std=0; 
		sscanf(param+15, "%i", &divide_by_std);

		if (divide_by_std)
			CIO::message(M_INFO, "normalizing VARIANCE\n");
		else
			CIO::message(M_WARN, "NOT normalizing VARIANCE\n");

		preproc=new CPruneVarSubMean(divide_by_std==1);
	}
	else 
	{
		CIO::not_implemented();
		return false;
	}

	preprocs->get_last_element();
	return preprocs->append_element(preproc);
}

bool CGUIPreProc::clean_preproc(CHAR* param)
{
	delete preprocs;
	preprocs=new CList<CPreProc*>(true);
	return (preprocs!=NULL);
}

bool CGUIPreProc::del_preproc(CHAR* param)
{
	CIO::message(M_INFO, "deleting preproc %i/(%i)\n", preprocs->get_num_elements()-1, preprocs->get_num_elements());


	CPreProc* p=preprocs->delete_element();
	if (p)
		delete p;
	return (p!=NULL);
}

bool CGUIPreProc::load(CHAR* param)
{
	bool result=false;

	param=CIO::skip_spaces(param);

	CPreProc* preproc=NULL;

	FILE* file=fopen(param, "r");
	CHAR id[5]="UDEF";

	if (file)
	{
		assert(fread(id, sizeof(char), 4, file)==4);
	
#ifdef HAVE_ATLAS
#ifdef HAVE_LAPACK
		if (strncmp(id, "PCAC", 4)==0)
		{
			preproc=new CPCACut();
		}
		else 
#endif // LAPACK
#endif // ATLAS
		if (strncmp(id, "NRM1", 4)==0)
		{
			preproc=new CNormOne();
		}
		else if (strncmp(id, "PVSM", 4)==0)
		{
			preproc=new CPruneVarSubMean();
		}
		else
			CIO::message(M_ERROR, "unrecognized file\n");

		if (preproc && preproc->load_init_data(file))
		{
			printf("file successfully read\n");
			result=true;
		}

		fclose(file);
	}
	else
		CIO::message(M_ERROR, "opening file %s failed\n", param);

	if (result)
	{
		preprocs->get_last_element();
		result=preprocs->append_element(preproc);
	}

	return result;
}

bool CGUIPreProc::save(CHAR* param)
{
	CHAR fname[1024];
	INT num=preprocs->get_num_elements()-1;
	bool result=false; param=CIO::skip_spaces(param);
	sscanf(param, "%s %i", fname, &num);
	CPreProc* preproc= preprocs->get_last_element();

	if (num>=0 && (num < preprocs->get_num_elements()) && preproc)
	{
		FILE* file=fopen(fname, "w");
	
		fwrite(preproc->get_id(), sizeof(char), 4, file);
		if ((!file) ||	(!preproc->save_init_data(file)))
			printf("writing to file %s failed!\n", param);
		else
		{
			printf("successfully written preproc init data into \"%s\" !\n", param);
			result=true;
		}

		if (file)
			fclose(file);
	}
	else
		CIO::message(M_ERROR, "create preproc first\n");

	return result;
}

bool CGUIPreProc::attach_preproc(CHAR* param)
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
				CFeatures* f = gui->guifeatures.get_train_features();
				if (f->get_feature_class()==C_COMBINED)
					f=((CCombinedFeatures*)f)->get_last_feature_obj();

				preprocess_features(f, NULL, force==1);
				gui->guifeatures.invalidate_train();
				result=true;
			}
			else if (strcmp(target,"TEST")==0)
			{
				CFeatures* fe = gui->guifeatures.get_test_features();
				CFeatures* f  = gui->guifeatures.get_train_features();

				if ((f->get_feature_class()==C_COMBINED) && 
					(fe->get_feature_class()==C_COMBINED))
					((CCombinedFeatures*)fe)->get_last_feature_pair(fe,f);
				
				preprocess_features(f, fe, force==1);
				gui->guifeatures.invalidate_test();
				result=true;
			}
			else
				CIO::message(M_ERROR, "see help for parameters\n");
		}
		else
			CIO::message(M_ERROR, "features not correctly assigned!\n");
	}
	else
		CIO::message(M_ERROR, "see help for parameters\n");

	/// when successful add preprocs to attached_preprocs list (for removal later)
	/// and clean the current preproc list
	if (result)
	{
		attached_preprocs_lists->get_last_element();
		attached_preprocs_lists->append_element(preprocs);
		preprocs=NULL;
	}

	return result;
}

bool CGUIPreProc::preprocess_features(CFeatures* trainfeat, CFeatures* testfeat, bool force)
{
	INT num_preproc=preprocs->get_num_elements();
	CPreProc** preprocs = NULL ;
	if (num_preproc > 0)
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

bool CGUIPreProc::preproc_all_features(CFeatures* f, bool force)
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
		case C_STRING:
			switch (f->get_feature_type())
			{
				case F_WORD:
					return ((CStringFeatures<WORD>*) f)->preproc_feature_strings(force);
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
