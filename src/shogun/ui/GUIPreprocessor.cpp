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

#include <shogun/ui/GUIPreprocessor.h>
#include <shogun/ui/SGInterface.h>

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/config.h>
#include <shogun/preprocessor/LogPlusOne.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/preprocessor/PruneVarSubMean.h>
#include <shogun/preprocessor/PCA.h>
#include <shogun/preprocessor/DecompressString.h>
#include <shogun/preprocessor/SortWordString.h>
#include <shogun/preprocessor/SortUlongString.h>
#include <shogun/features/RealFileFeatures.h>
#include <shogun/features/TOPFeatures.h>
#include <shogun/features/FKFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/Features.h>

#include <string.h>
#include <stdio.h>

using namespace shogun;

CGUIPreprocessor::CGUIPreprocessor(CSGInterface* ui_)
: CSGObject(), ui(ui_)
{
	preprocs=new CList(true);
}

CGUIPreprocessor::~CGUIPreprocessor()
{
	SG_UNREF(preprocs);
}

CPreprocessor* CGUIPreprocessor::create_prunevarsubmean(bool divide_by_std)
{
	CPreprocessor* preproc=new CPruneVarSubMean(divide_by_std);

	if (preproc)
		SG_INFO("PRUNEVARSUBMEAN created (%p), divide_by_std %d", preproc, divide_by_std)
	else
		SG_ERROR("Could not create preproc PRUNEVARSUBMEAN, divide_by_std %d", divide_by_std)

	return preproc;
}

CPreprocessor* CGUIPreprocessor::create_pca(bool do_whitening, float64_t threshold)
{
#ifdef HAVE_EIGEN3
	CPreprocessor* preproc=new CPCA(do_whitening, THRESHOLD, threshold);

	if (preproc)
		SG_INFO("PCA created (%p), do_whitening %i threshold %e", preproc, do_whitening, threshold)
	else
		SG_ERROR("Could not create preproc PCA, do_whitening %i threshold %e", do_whitening, threshold)

	return preproc;
#else //HAVE_EIGEN3
	SG_ERROR("Could not create preproc PCA - eigen3 not available at compile time\n")
	return NULL;
#endif //HAVE_EIGEN3
}

CPreprocessor* CGUIPreprocessor::create_generic(EPreprocessorType type)
{
	CPreprocessor* preproc=NULL;

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
		case P_DECOMPRESSCHARSTRING:
			preproc=new CDecompressString<char>(LZO); break;
		default:
			SG_ERROR("Unknown Preprocessor type %d\n", type)
	}

	if (preproc)
		SG_INFO("Preproc of type %d created (%p).\n", type, preproc)
	else
		SG_ERROR("Could not create preproc of type %d.\n", type)

	return preproc;
}

bool CGUIPreprocessor::add_preproc(CPreprocessor* preproc)
{
	return preprocs->append_element_at_listend(preproc);
}

bool CGUIPreprocessor::clean_preproc()
{
	SG_UNREF(preprocs);
	preprocs=new CList(true);
	return (preprocs!=NULL);
}

bool CGUIPreprocessor::del_preproc()
{
	SG_INFO("Deleting preproc %i/(%i).\n", preprocs->get_num_elements()-1, preprocs->get_num_elements())

	CSGObject* preproc=preprocs->delete_element();
	SG_UNREF(preproc);

	return (preproc!=NULL);
}

bool CGUIPreprocessor::attach_preproc(char* target, bool do_force)
{
	bool result=false;

	if (strncmp(target, "TRAIN", 5)==0)
	{
		CFeatures* f=ui->ui_features->get_train_features();
		if (!f)
			SG_ERROR("No train features assigned!\n")

		if (f->get_feature_class()==C_COMBINED)
			f=((CCombinedFeatures*)f)->get_last_feature_obj();

		preprocess_features(f, NULL, do_force);
		ui->ui_features->invalidate_train();
		result=true;
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		CFeatures* f_test=ui->ui_features->get_test_features();
		if (!f_test)
			SG_ERROR("No test features assigned!\n")

		CFeatures* f_train=ui->ui_features->get_train_features();
		if (!f_train)
			SG_ERROR("No train features assigned!\n")

		EFeatureClass fclass_train=f_train->get_feature_class();
		EFeatureClass fclass_test=f_test->get_feature_class();

		if (fclass_train==fclass_test)
		{
			if (fclass_train==C_COMBINED)
			{
				if (((CCombinedFeatures*) f_train)->check_feature_obj_compatibility((CCombinedFeatures*) f_test))
				{

					int32_t num_combined=((CCombinedFeatures*) f_test)->get_num_feature_obj();
					ASSERT(((CCombinedFeatures*) f_train)->get_num_feature_obj()==num_combined)

					if (!num_combined)
						SG_ERROR("One of the combined features has no sub-features ?!\n")

					//preprocess the last test feature obj
					SG_INFO("BEGIN PREPROCESSING COMBINED FEATURES (%d sub-featureobjects).\n", num_combined)
					index_t f_idx = 0;
					for (; f_idx<num_combined; f_idx++)
					{
						CFeatures* te_feat=((CCombinedFeatures*) f_test)->get_feature_obj(f_idx);
						CFeatures* tr_feat=((CCombinedFeatures*) f_train)->get_feature_obj(f_idx);

						if (!(te_feat && tr_feat))
							break;

						// and preprocess using that one
						SG_INFO("TRAIN ")
						tr_feat->list_feature_obj();
						SG_INFO("TEST ")
						te_feat->list_feature_obj();
						preprocess_features(tr_feat, te_feat, do_force);
					}
					ASSERT(f_idx==num_combined)
					result=true;
					SG_INFO("END PREPROCESSING COMBINED FEATURES\n")
				}
				else
					SG_ERROR("combined features not compatible\n")
			}
			else
			{
				preprocess_features(f_train, f_test, do_force);
				ui->ui_features->invalidate_test();
				result=true;
			}
		}
		else
			SG_ERROR("Features not compatible.\n")
	}
	else
		SG_ERROR("Features not correctly assigned!\n")

	/// when successful create new preproc list
	if (result)
		clean_preproc();

	return result;
}

bool CGUIPreprocessor::preprocess_features(CFeatures* trainfeat, CFeatures* testfeat, bool force)
{
	if (trainfeat)
	{
		if (testfeat)
		{
			// if we don't have a preproc for trainfeatures we
			// don't need a preproc for test features
			SG_DEBUG("%d preprocessors attached to train features %d to test features\n", trainfeat->get_num_preprocessors(), testfeat->get_num_preprocessors())

			if (trainfeat->get_num_preprocessors() < testfeat->get_num_preprocessors())
			{
				SG_ERROR("more preprocessors attached to test features than to train features\n")
				return false;
			}

			if (trainfeat->get_num_preprocessors() && (trainfeat->get_num_preprocessors() > testfeat->get_num_preprocessors()))
			{
				for (int32_t i=0; i<trainfeat->get_num_preprocessors();  i++)
				{
					CPreprocessor* preproc = trainfeat->get_preprocessor(i);
					preproc->init(trainfeat);
					testfeat->add_preprocessor(preproc);
					SG_UNREF(preproc);
				}

				preproc_all_features(testfeat, force);
			}
		}
		else
		{
			CPreprocessor* preproc = (CPreprocessor*) preprocs->get_first_element();

			if (preproc)
			{
				preproc->init(trainfeat);
				trainfeat->add_preprocessor(preproc);

				preproc_all_features(trainfeat, force);
				SG_UNREF(preproc);
			}

			while ( (preproc = (CPreprocessor*) preprocs->get_next_element()) !=NULL )
			{
				preproc->init(trainfeat);
				trainfeat->add_preprocessor(preproc);
				SG_UNREF(preproc);

				preproc_all_features(trainfeat, force);
			}
		}

		return true;
	}
	else
		SG_ERROR("no features for preprocessing available!\n")

	return false;
}

bool CGUIPreprocessor::preproc_all_features(CFeatures* f, bool force)
{
	switch (f->get_feature_class())
	{
		case C_DENSE:
			switch (f->get_feature_type())
			{
				case F_DREAL:
					return ((CDenseFeatures<float64_t>*) f)->apply_preprocessor(force);
				case F_SHORT:
					return ((CDenseFeatures<int16_t>*) f)->apply_preprocessor(force);
				case F_WORD:
					return ((CDenseFeatures<uint16_t>*) f)->apply_preprocessor(force);
				case F_CHAR:
					return ((CDenseFeatures<char>*) f)->apply_preprocessor(force);
				case F_BYTE:
					return ((CDenseFeatures<uint8_t>*) f)->apply_preprocessor(force);
				default:
					SG_NOTIMPLEMENTED
			}
			break;
		case C_STRING:
			switch (f->get_feature_type())
			{
				case F_WORD:
					return ((CStringFeatures<uint16_t>*) f)->apply_preprocessor(force);
				case F_ULONG:
					return ((CStringFeatures<uint64_t>*) f)->apply_preprocessor(force);
				default:
					SG_NOTIMPLEMENTED
			}
			break;
		case C_SPARSE:
			switch (f->get_feature_type())
			{
				case F_DREAL:
					return ((CSparseFeatures<float64_t>*) f)->apply_preprocessor(force);
				default:
					SG_NOTIMPLEMENTED
			};
			break;
		case C_COMBINED:
			SG_ERROR("Combined feature objects cannot be preprocessed. Only its sub-feature objects!\n")
			break;
		default:
			SG_NOTIMPLEMENTED
	}

	return false;
}
