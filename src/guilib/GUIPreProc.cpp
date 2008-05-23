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
#include "lib/io.h"
#include "lib/config.h"

#include "guilib/GUIPreProc.h"
#include "interface/SGInterface.h"

#include "preproc/LogPlusOne.h"
#include "preproc/NormOne.h"
#include "preproc/PruneVarSubMean.h"
#include "preproc/PCACut.h"
#include "preproc/SortWord.h"
#include "preproc/SortWordString.h"
#include "preproc/SortUlongString.h"
#include "features/RealFileFeatures.h"
#include "features/TOPFeatures.h"
#include "features/FKFeatures.h"
#include "features/CharFeatures.h"
#include "features/StringFeatures.h"
#include "features/ByteFeatures.h"
#include "features/ShortFeatures.h"
#include "features/RealFeatures.h"
#include "features/SparseFeatures.h"
#include "features/CombinedFeatures.h"
#include "features/Features.h"

#include <string.h>
#include <stdio.h>

CGUIPreProc::CGUIPreProc(CSGInterface* ui_)
  : CSGObject(), ui(ui_)
{
	preprocs=new CList<CPreProc*>(true);
	attached_preprocs_lists=new CList<CList<CPreProc*>*>(true);
}

CGUIPreProc::~CGUIPreProc()
{
	delete preprocs;
	delete attached_preprocs_lists;
}

CPreProc* CGUIPreProc::create_prunevarsubmean(bool divide_by_std)
{
	CPreProc* preproc=new CPruneVarSubMean(divide_by_std);

	if (preproc)
		SG_INFO("PRUNEVARSUBMEAN created (%p), divide_by_std %d", preproc, divide_by_std);
	else
		SG_ERROR("Could not create preproc PRUNEVARSUBMEAN, divide_by_std %d", divide_by_std);

	return preproc;
}

CPreProc* CGUIPreProc::create_pcacut(bool do_whitening, DREAL threshold)
{
#ifdef HAVE_LAPACK
	CPreProc* preproc=new CPCACut(do_whitening, threshold);

	if (preproc)
		SG_INFO("PCACUT created (%p), do_whitening %i threshold %e", preproc, do_whitening, threshold);
	else
		SG_ERROR("Could not create preproc PCACUT, do_whitening %i threshold %e", do_whitening, threshold);

	return preproc;
#else //HAVE_LAPACK
	SG_ERROR("Could not create preproc PCACUT - lapack not available at compile time\n");
	return NULL;
#endif //HAVE_LAPACK
}

CPreProc* CGUIPreProc::create_generic(EPreProcType type)
{
	CPreProc* preproc=NULL;

	switch (type)
	{
		case P_NORMONE:
			preproc=new CNormOne(); break;
		case P_LOGPLUSONE:
			preproc=new CLogPlusOne(); break;
		case P_SORTWORDSTRING:
			preproc=new CSortWordString(); break;
		case P_SORTULONGSTRING:
			preproc=new CSortUlongString(); break;
		case P_SORTWORD:
			preproc=new CSortWord(); break;
		default:
			SG_ERROR("Unknown PreProc type %d\n", type);
	}

	if (preproc)
		SG_INFO("Preproc of type %d created (%p).\n", type, preproc);
	else
		SG_ERROR("Could not create preproc of type %d.\n", type);

	return preproc;
}

bool CGUIPreProc::add_preproc(CPreProc* preproc)
{
	preprocs->get_last_element();
	return preprocs->append_element(preproc);
}

bool CGUIPreProc::clean_preproc()
{
	delete preprocs;
	preprocs=new CList<CPreProc*>(true);
	return (preprocs!=NULL);
}

bool CGUIPreProc::del_preproc()
{
	SG_INFO("Deleting preproc %i/(%i).\n", preprocs->get_num_elements()-1, preprocs->get_num_elements());

	CPreProc* preproc=preprocs->delete_element();
	if (preproc)
		delete preproc;

	return (preproc!=NULL);
}

bool CGUIPreProc::load(CHAR* filename)
{
	bool result=false;
	CPreProc* preproc=NULL;

	FILE* file=fopen(filename, "r");
	CHAR id[5]="UDEF";

	if (file)
	{
		ASSERT(fread(id, sizeof(char), 4, file)==4);
	
#ifdef HAVE_LAPACK
		if (strncmp(id, "PCAC", 4)==0)
		{
			preproc=new CPCACut();
		}
		else 
#endif
		if (strncmp(id, "NRM1", 4)==0)
		{
			preproc=new CNormOne();
		}
		else if (strncmp(id, "PVSM", 4)==0)
		{
			preproc=new CPruneVarSubMean();
		}
		else
			SG_ERROR("Unrecognized file %s.\n", filename);

		if (preproc && preproc->load_init_data(file))
		{
			printf("File %s successfully read.\n", filename);
			result=true;
		}

		fclose(file);
	}
	else
		SG_ERROR("Opening file %s failed\n", filename);

	if (result)
	{
		preprocs->get_last_element();
		result=preprocs->append_element(preproc);
	}

	return result;
}

bool CGUIPreProc::save(CHAR* filename, INT num_preprocs)
{
	bool result=false;
	CPreProc* preproc=preprocs->get_last_element();

	INT num=preprocs->get_num_elements()-1;
	if (num_preprocs>=0)
		num=num_preprocs;

	if (num>=0 && num<preprocs->get_num_elements() && preproc)
	{
		FILE* file=fopen(filename, "w");
	
		fwrite(preproc->get_id(), sizeof(char), 4, file);
		if (!file || !preproc->save_init_data(file))
			printf("Writing to file %s failed!\n", filename);
		else
		{
			SG_INFO("Successfully written preproc init data into %s!\n", filename);
			result=true;
		}

		if (file)
			fclose(file);
	}
	else
		SG_ERROR("Create at least one preproc first.\n");

	return result;
}

bool CGUIPreProc::attach_preproc(CHAR* target, bool do_force)
{
	bool result=false;

	if (strncmp(target, "TRAIN", 5)==0)
	{
		CFeatures* f=ui->ui_features->get_train_features();
		if (f->get_feature_class()==C_COMBINED)
			f=((CCombinedFeatures*)f)->get_last_feature_obj();

		preprocess_features(f, NULL, do_force);
		ui->ui_features->invalidate_train();
		result=true;
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		CFeatures* f_test=ui->ui_features->get_test_features();
		CFeatures* f_train=ui->ui_features->get_train_features();
		EFeatureClass fclass_train=f_train->get_feature_class();
		EFeatureClass fclass_test=f_test->get_feature_class();

		if (fclass_train==fclass_test)
		{
			if (fclass_train==C_COMBINED)
			{
				if (((CCombinedFeatures*) f_train)->check_feature_obj_compatibility((CCombinedFeatures*) f_test))
				{
					//preprocess the last test feature obj
					CFeatures* te_feat=((CCombinedFeatures*) f_test)->get_first_feature_obj();
					CFeatures* tr_feat=((CCombinedFeatures*) f_train)->get_first_feature_obj();

					INT num_combined=((CCombinedFeatures*) f_test)->get_num_feature_obj();
					ASSERT(((CCombinedFeatures*) f_train)->get_num_feature_obj()==num_combined);

					if (!(num_combined && tr_feat && te_feat))
						SG_ERROR("One of the combined features has no sub-features ?!\n");

					SG_INFO("BEGIN PREPROCESSING COMBINED FEATURES (%d sub-featureobjects).\n", num_combined);
					
					int n=0;
					while (n<num_combined && tr_feat && te_feat)
					{
						// and preprocess using that one 
						SG_INFO("TRAIN ");
						tr_feat->list_feature_obj();
						SG_INFO("TEST ");
						te_feat->list_feature_obj();
						preprocess_features(tr_feat, te_feat, do_force);
						tr_feat=((CCombinedFeatures*) f_train)->get_next_feature_obj();
						te_feat=((CCombinedFeatures*) f_test)->get_next_feature_obj();
						n++;
					}
					ASSERT(n==num_combined);
					result=true;
					SG_INFO( "END PREPROCESSING COMBINED FEATURES\n");
				}
				else
					SG_ERROR( "combined features not compatible\n");
			}
			else
			{
				preprocess_features(f_train, f_test, do_force);
				ui->ui_features->invalidate_test();
				result=true;
			}
		}
		else
			SG_ERROR("Features not compatible.\n");
	}
	else
		SG_ERROR("Features not correctly assigned!\n");

	/// when successful add preprocs to attached_preprocs list (for removal later)
	/// and clean the current preproc list
	if (result)
	{
		attached_preprocs_lists->get_last_element();
		attached_preprocs_lists->append_element(preprocs);
		preprocs=new CList<CPreProc*>(true);
	}

	return result;
}

bool CGUIPreProc::preprocess_features(CFeatures* trainfeat, CFeatures* testfeat, bool force)
{
	if (trainfeat)
	{
		if (testfeat)
		{
			// if we don't have a preproc for trainfeatures we 
			// don't need a preproc for test features
			SG_DEBUG( "%d preprocessors attached to train features %d to test features\n", trainfeat->get_num_preproc(), testfeat->get_num_preproc());

			if (trainfeat->get_num_preproc() < testfeat->get_num_preproc())
			{
				SG_ERROR( "more preprocessors attached to test features than to train features\n");
				return false;
			}

			if (trainfeat->get_num_preproc() && (trainfeat->get_num_preproc() > testfeat->get_num_preproc()))
			{
				for (INT i=0; i<trainfeat->get_num_preproc();  i++)
				{
					CPreProc* preproc = trainfeat->get_preproc(i);
					preproc->init(trainfeat);
					testfeat->add_preproc(trainfeat->get_preproc(i));
				}

				preproc_all_features(testfeat, force);
			}
		}
		else
		{
			CPreProc* preproc = preprocs->get_first_element();

			if (preproc)
			{
				preproc->init(trainfeat);
				trainfeat->add_preproc(preproc);

				preproc_all_features(trainfeat, force);
			}

			while ( (preproc = preprocs->get_next_element()) !=NULL )
			{
				preproc->init(trainfeat);
				trainfeat->add_preproc(preproc);

				preproc_all_features(trainfeat, force);
			}
		}

		return true;
	}
	else
		SG_ERROR( "no features for preprocessing available!\n");

	return false;
}

bool CGUIPreProc::preproc_all_features(CFeatures* f, bool force)
{
	switch (f->get_feature_class())
	{
		case C_SIMPLE:
			switch (f->get_feature_type())
			{
				case F_DREAL:
					return ((CRealFeatures*) f)->apply_preproc(force);
				case F_SHORT:
					return ((CShortFeatures*) f)->apply_preproc(force);
				case F_WORD:
					return ((CShortFeatures*) f)->apply_preproc(force);
				case F_CHAR:
					return ((CCharFeatures*) f)->apply_preproc(force);
				case F_BYTE:
					return ((CByteFeatures*) f)->apply_preproc(force);
				default:
					io.not_implemented();
			}
			break;
		case C_STRING:
			switch (f->get_feature_type())
			{
				case F_WORD:
					return ((CStringFeatures<WORD>*) f)->apply_preproc(force);
				case F_ULONG:
					return ((CStringFeatures<ULONG>*) f)->apply_preproc(force);
				default:
					io.not_implemented();
			}
			break;
		case C_SPARSE:
			switch (f->get_feature_type())
			{
				case F_DREAL:
					return ((CSparseFeatures<DREAL>*) f)->apply_preproc(force);
				default:
					io.not_implemented();
			};
			break;
		case C_COMBINED:
			SG_ERROR( "Combined feature objects cannot be preprocessed. Only its sub-feature objects!\n");
			break;
		default:
			io.not_implemented();
	}

	return false;
}
#endif
