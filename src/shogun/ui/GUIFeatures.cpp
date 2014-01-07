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

#include <ui/GUIFeatures.h>
#include <ui/SGInterface.h>

#include <lib/config.h>
#include <io/SGIO.h>
#include <io/CSVFile.h>

using namespace shogun;

CGUIFeatures::CGUIFeatures(CSGInterface* ui_)
: CSGObject(), ui(ui_), train_features(NULL), test_features(NULL),
	ref_features(NULL)
{
}

CGUIFeatures::~CGUIFeatures()
{
	SG_UNREF(train_features);
	SG_UNREF(test_features);
	SG_UNREF(ref_features);
}

void CGUIFeatures::invalidate_train()
{
	CKernel *k = ui->ui_kernel->get_kernel();
	if (k)
		k->remove_lhs();
}

void CGUIFeatures::invalidate_test()
{
	CKernel *k = ui->ui_kernel->get_kernel();
	if (k)
		k->remove_rhs();
}

bool CGUIFeatures::load(
	char* filename, char* fclass, char* type, char* target, int32_t size,
	int32_t comp_features)
{
	bool result=false;
	CFeatures** f_ptr=NULL;

	if (strncmp(target, "TRAIN", 5)==0)
	{
		f_ptr=&train_features;
		invalidate_train();
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		f_ptr=&test_features;
		invalidate_test();
	}
	else
		SG_ERROR("Unknown target %s, neither TRAIN nor TEST.\n", target)

	SG_UNREF(*f_ptr);
	*f_ptr=NULL;

	CCSVFile* file=new CCSVFile(filename);
	if (strncmp(fclass, "SIMPLE", 6)==0)
	{
		if (strncmp(type, "REAL", 4)==0)
		{
			*f_ptr=new CDenseFeatures<float64_t>(file);
		}
		else if (strncmp(type, "BYTE", 4)==0)
		{
			///FIXME make CHAR type configurable... it is DNA by default
			*f_ptr=new CDenseFeatures<uint8_t>(file);
		}
		else if (strncmp(type, "CHAR", 4)==0)
		{
			///FIXME make CHAR type configurable... it is DNA by default
			*f_ptr=new CDenseFeatures<char>(file);
		}
		else if (strncmp(type, "SHORT", 5)==0)
		{
			*f_ptr=new CDenseFeatures<int16_t>(file);
		}
		else
		{
			SG_ERROR("Unknown type.\n")
			return false;
		}
	}
	else if (strncmp(fclass, "SPARSE", 6)==0)
	{
		SG_NOTIMPLEMENTED
	}
	else if (strncmp(fclass, "STRING", 6)==0)
	{
		if (strncmp(type, "REAL", 4)==0)
		{
			*f_ptr=new CStringFeatures<float64_t>(file);
		}
		else if (strncmp(type, "BYTE", 4)==0)
		{
			///FIXME make CHAR type configurable... it is DNA by default
			*f_ptr=new CStringFeatures<uint8_t>(file, DNA);
		}
		else if (strncmp(type, "CHAR", 4)==0)
		{
			///FIXME make CHAR type configurable... it is DNA by default
			*f_ptr=new CStringFeatures<char>(file, DNA);
		}
		else if (strncmp(type, "SHORT", 5)==0)
		{
			*f_ptr=new CStringFeatures<int16_t>(file);
		}
		else if (strncmp(type, "WORD", 4)==0)
		{
			*f_ptr=new CStringFeatures<uint16_t>(file);
		}
		else if (strncmp(type, "ULONG", 5)==0)
		{
			*f_ptr=new CStringFeatures<uint64_t>(file);
		}
		else
		{
			SG_ERROR("Unknown type.\n")
			return false;
		}
	}
	SG_UNREF(file);

	return result;
}

bool CGUIFeatures::save(char* filename, char* type, char* target)
{
	bool result=false;

	CFeatures** f_ptr=NULL;

	if (strncmp(target, "TRAIN", 5)==0)
	{
		f_ptr=&train_features;
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		f_ptr=&test_features;
	}
	else
		SG_ERROR("Unknown target %s, neither TRAIN nor TEST.\n", target)

	if (*f_ptr)
	{
		try
		{
			CCSVFile* file=new CCSVFile(filename, 'w');
			if (strncmp(type, "REAL", 4)==0)
			{
				((CDenseFeatures<float64_t>*) (*f_ptr))->save(file);
			}
			else if (strncmp(type, "BYTE", 4)==0)
			{
				((CDenseFeatures<uint8_t>*) (*f_ptr))->save(file);
			}
			else if (strncmp(type, "CHAR", 4)==0)
			{
				((CDenseFeatures<char>*) (*f_ptr))->save(file);
			}
			else if (strncmp(type, "SHORT", 5)==0)
			{
				((CDenseFeatures<int16_t>*) (*f_ptr))->save(file);
			}
			else if (strncmp(type, "WORD", 4)==0)
			{
				((CDenseFeatures<uint16_t>*) (*f_ptr))->save(file);
			}
			else
			{
				SG_ERROR("Unknown type.\n")
				return false;
			}
			SG_UNREF(file);
		}
		catch (...)
		{
			SG_ERROR("Writing to file %s failed!\n", filename)
		}

		SG_INFO("Successfully written features into \"%s\" !\n", filename)
		result=true;

	} else
		SG_ERROR("Set features first.\n")

	return result;
}

bool CGUIFeatures::clean(char* target)
{
	if (strncmp(target, "TRAIN", 5)==0)
		set_train_features(NULL);
	else if (strncmp(target, "TEST", 4)==0)
		set_test_features(NULL);
	else
		SG_ERROR("Unknown target %s, neither TRAIN nor TEST.\n", target)

	return true;
}

bool CGUIFeatures::reshape(char* target, int32_t num_feat, int32_t num_vec)
{
	CFeatures** f_ptr=NULL;

	if (strncmp(target, "TRAIN", 5)==0)
	{
		f_ptr=&train_features;
		invalidate_train();
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		f_ptr=&test_features;
		invalidate_test();
	}
	else
	{
		SG_ERROR("Invalid target %s\n", target)
		return false;
	}

	bool result=false;
	if (f_ptr)
	{
		SG_INFO("reshape data to %d x %d\n", num_feat, num_vec)
		result=(*f_ptr)->reshape(num_feat, num_vec);

		if (!result)
			SG_ERROR("Reshaping failed.\n")
	}

	return result;
}

CFeatures* CGUIFeatures::get_convert_features(char* target)
{
	CFeatures* features;

	if (strncmp(target, "TEST", 4)==0)
		features=get_test_features();
	else if (strncmp(target, "TRAIN", 5)==0)
		features=get_train_features();
	else
		return NULL;

	if (features->get_feature_class()==C_COMBINED)
		features=((CCombinedFeatures*) features)->get_last_feature_obj();

	return features;
}

bool CGUIFeatures::set_convert_features(CFeatures* features, char* target)
{
	CFeatures* features_prev;

	if (strncmp(target, "TEST", 4)==0)
		features_prev=get_test_features();
	else if (strncmp(target, "TRAIN", 5)==0)
		features_prev=get_train_features();
	else
		return false;

	// in case of combined features delete current (==last) feature obj
	// pointer from list (feature object got deleted already above)
	// and append *f_ptr which holds the newly created feature object
	if (features_prev->get_feature_class()==C_COMBINED)
	{
		CCombinedFeatures* combined=(CCombinedFeatures*) features_prev;
		combined->delete_feature_obj(combined->get_num_feature_obj()-1);
		combined->append_feature_obj(features);
		combined->list_feature_objs();
	}
	else // set features to new test/train features
	{
		if (strncmp(target, "TEST", 4)==0)
			set_test_features(features);
		else
			set_train_features(features);
	}

	return true;
}

CSparseFeatures<float64_t>* CGUIFeatures::convert_simple_real_to_sparse_real(
	CDenseFeatures<float64_t>* src)
{
	if (src &&
		src->get_feature_class()==C_DENSE &&
		src->get_feature_type()==F_DREAL)
	{
		//create sparse features with 0 cache
		SG_INFO("Attempting to convert dense feature matrix to a sparse one.\n")
		CSparseFeatures<float64_t>* target=new CSparseFeatures<float64_t>(0);
		int32_t num_f=0;
		int32_t num_v=0;
		float64_t* feats=src->get_feature_matrix(num_f, num_v);
		target->set_full_feature_matrix(SGMatrix<float64_t>(feats, num_f, num_v));
		return target;
	}
	else
		SG_ERROR("No SIMPLE DREAL features available.\n")

	return NULL;
}

CStringFeatures<char>* CGUIFeatures::convert_simple_char_to_string_char(
	CDenseFeatures<char>* src)
{
	if (src && src->get_feature_class()==C_DENSE)
	{
		int32_t num_vec=src->get_num_vectors();
		SGString<char>* strings=SG_MALLOC(SGString<char>, num_vec);
		int32_t max_len=-1;

		for (int32_t i=0; i<num_vec; i++)
		{
			bool to_free=false;
			int32_t len=0;
			char* str=src->get_feature_vector(i, len, to_free);
			strings[i].slen=len ;
			for (int32_t j=0; j<len; j++)
				if (str[j]==0)
				{
					strings[i].slen=j ;
					break ;
				} ;
			strings[i].string=SG_MALLOC(char, strings[i].slen);

			for (int32_t j=0; j<strings[i].slen; j++)
				strings[i].string[j]=str[j];

			if (strings[i].slen> max_len)
				max_len=strings[i].slen;

			src->free_feature_vector(str, i, to_free);
		}

		CStringFeatures<char>* target=new CStringFeatures<char>(new CAlphabet(DNA));
		target->set_features(strings, num_vec, max_len);
		return target;
	}
	else
		SG_ERROR("No features of class/type SIMPLE/CHAR available.\n")

	return NULL;
}

CDenseFeatures<float64_t>* CGUIFeatures::convert_simple_word_to_simple_salzberg(
	CDenseFeatures<uint16_t>* src)
{
	CPluginEstimate* pie=ui->ui_pluginestimate->get_estimator();

	if (src &&
		src->get_feature_type()==F_WORD &&
		src->get_feature_class()==C_DENSE &&
		pie)
	{
		CDenseFeatures<float64_t>* target=new CDenseFeatures<float64_t>(0);
		int32_t num_feat=src->get_num_features();
		int32_t num_vec=src->get_num_vectors();
		float64_t* fm=SG_MALLOC(float64_t, num_vec*num_feat);

		if (fm)
		{
			for (int32_t i=0; i<num_vec; i++)
			{
				int32_t len=0;
				bool to_free=false;
				uint16_t* vec = src->get_feature_vector(i, len, to_free);
				ASSERT(num_feat==len)

				for (int32_t j=0; j<num_feat; j++)
					fm[i*num_feat+j]=
						pie->get_parameterwise_log_odds(vec[j], j);

				src->free_feature_vector(vec, i, to_free);
			}
			target->set_feature_matrix(SGMatrix<float64_t>(fm, num_feat, num_vec));

		}
		return target;
	}
	else
		SG_ERROR("No SIMPLE WORD features or PluginEstimator available.\n")

	return NULL;
}


CTOPFeatures* CGUIFeatures::convert_string_word_to_simple_top(
	CStringFeatures<uint16_t>* src)
{
	CTOPFeatures* tf=NULL;

	if (src &&
		src->get_feature_class()==C_DENSE &&
		src->get_feature_type()==F_WORD)
	{
		SG_INFO("Converting to TOP features.\n")

		if (ui->ui_hmm->get_pos() && ui->ui_hmm->get_neg())
		{
			ui->ui_hmm->get_pos()->set_observations(src);
			ui->ui_hmm->get_neg()->set_observations(src);

			bool neglinear=false;
			bool poslinear=false;

			tf=new CTOPFeatures(
				0, ui->ui_hmm->get_pos(), ui->ui_hmm->get_neg(),
				neglinear, poslinear);
			ASSERT(tf->set_feature_matrix())
		}
		else
			SG_ERROR("HMMs not correctly assigned!\n")
	}
	else
		SG_ERROR("No SIMPLE WORD features available.\n")

	return tf;
}

CFKFeatures* CGUIFeatures::convert_string_word_to_simple_fk(
	CStringFeatures<uint16_t>* src)
{
	CFKFeatures* fkf=NULL;

	SG_INFO("Converting to FK features.\n")

	if (ui->ui_hmm->get_pos() && ui->ui_hmm->get_neg())
	{
		CStringFeatures<uint16_t>* old_obs_pos=
			ui->ui_hmm->get_pos()->get_observations();
		CStringFeatures<uint16_t>* old_obs_neg=
			ui->ui_hmm->get_neg()->get_observations();

		CStringFeatures<uint16_t>* string_feat=src;
		ui->ui_hmm->get_pos()->set_observations(string_feat);
		ui->ui_hmm->get_neg()->set_observations(string_feat);

		fkf=new CFKFeatures(
			0, ui->ui_hmm->get_pos(), ui->ui_hmm->get_neg());
			//, neglinear, poslinear);
		if (train_features)
			fkf->set_opt_a(((CFKFeatures*) train_features)->get_weight_a());
		else
			SG_ERROR("Need train features to set optimal a.\n")

		ASSERT(fkf->set_feature_matrix())

		ui->ui_hmm->get_pos()->set_observations(old_obs_pos);
		ui->ui_hmm->get_neg()->set_observations(old_obs_neg);
	}
	else
		SG_ERROR("HMMs not correctly assigned!\n")

	return fkf;
}


CDenseFeatures<float64_t>* CGUIFeatures::convert_sparse_real_to_simple_real(
	CSparseFeatures<float64_t>* src)
{
	if (src &&
		src->get_feature_class()==C_SPARSE &&
		src->get_feature_type() == F_DREAL)
	{
		//create dense features with 0 cache
		SG_INFO("Attempting to convert sparse feature matrix to a dense one.\n")
		CDenseFeatures<float64_t>* rf=new CDenseFeatures<float64_t>(0);
		if (rf)
		{
			SGMatrix<float64_t> feats=src->get_full_feature_matrix();
			rf->set_feature_matrix(feats);
			return rf;
		}
	}
	else
		SG_ERROR("No SPARSE REAL features available.\n")

	return NULL;
}

CExplicitSpecFeatures* CGUIFeatures::convert_string_byte_to_spec_word(
		CStringFeatures<uint16_t>* src, bool use_norm)
{
	return new CExplicitSpecFeatures(src, use_norm);
}

CDenseFeatures<float64_t>* CGUIFeatures::convert_simple_char_to_simple_align(
	CDenseFeatures<char>* src, float64_t gap_cost)
{
	if (src &&
		src->get_feature_class()==C_DENSE &&
		src->get_feature_type()==F_CHAR)
	{
		//create dense features with 0 cache
		SG_INFO("Converting CHAR features to REAL ones.\n")

		CDenseFeatures<float64_t>* rf=new CDenseFeatures<float64_t>(0);
		if (rf)
		{
			SG_INFO("Start aligment with gapCost=%1.2f.\n", gap_cost)
			/*rf->Align_char_features(
				src, (CDenseFeatures<char>*) ref_features, gap_cost);*/
			SG_INFO("Conversion was successful.\n")
			return rf;
		}
	}
	else
		SG_ERROR("No SIMPLE CHAR features available.\n")

	SG_ERROR("Conversion failed.\n")
	return NULL;
}

bool CGUIFeatures::set_reference_features(char* target)
{
	if (strncmp(target, "TRAIN", 5)==0)
	{
		SG_UNREF(ref_features);
		ref_features=train_features;
		train_features=NULL;
		invalidate_train();
		return true;
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		SG_UNREF(ref_features);
		ref_features=test_features;
		test_features=NULL;
		invalidate_test();
		return true;
	}

	return false;
}

void CGUIFeatures::add_train_features(CFeatures* f)
{
	ASSERT(f)
	invalidate_train();

	if (!train_features)
	{
		train_features=new CCombinedFeatures();
		SG_REF(train_features);
	}

	if (train_features->get_feature_class()!=C_COMBINED)
	{
		CFeatures* first_elem=train_features;
		train_features=new CCombinedFeatures();
		SG_REF(train_features);
		((CCombinedFeatures*) train_features)->append_feature_obj(first_elem);
		((CCombinedFeatures*) train_features)->list_feature_objs();
		SG_UNREF(first_elem);
	}

	bool result=((CCombinedFeatures*) train_features)->append_feature_obj(f);
	if (result)
		((CCombinedFeatures*) train_features)->list_feature_objs();
	else
		SG_ERROR("appending feature object failed\n")
}

void CGUIFeatures::add_train_dotfeatures(CDotFeatures* f)
{
	ASSERT(f)
	SG_PRINT("DOTFVEC %d\n", f->get_num_vectors())
	invalidate_train();

	if (!train_features)
	{
		train_features=new CCombinedDotFeatures();
		SG_REF(train_features);
	}

	if (train_features->get_feature_class()!=C_COMBINED_DOT)
	{
		if (!train_features->has_property(FP_DOT))
			SG_ERROR("Trainfeatures not based on DotFeatures.\n")

		CDotFeatures* first_elem=(CDotFeatures*) train_features;
		train_features=new CCombinedDotFeatures();
		SG_REF(train_features);
		((CCombinedDotFeatures*) train_features)->append_feature_obj(first_elem);
		((CCombinedDotFeatures*) train_features)->list_feature_objs();
		SG_UNREF(first_elem);
	}

	bool result=((CCombinedDotFeatures*) train_features)->append_feature_obj(f);
	if (result)
		((CCombinedDotFeatures*) train_features)->list_feature_objs();
	else
		SG_ERROR("appending dot feature object failed\n")
}

void CGUIFeatures::add_test_dotfeatures(CDotFeatures* f)
{
	ASSERT(f)
	invalidate_test();

	if (!test_features)
	{
		test_features=new CCombinedDotFeatures();
		SG_REF(test_features);
	}

	if (test_features->get_feature_class()!=C_COMBINED_DOT)
	{
		if (!test_features->has_property(FP_DOT))
			SG_ERROR("Trainfeatures not based on DotFeatures.\n")

		CDotFeatures* first_elem=(CDotFeatures*) test_features;
		test_features=new CCombinedDotFeatures();
		SG_REF(test_features);
		((CCombinedDotFeatures*) test_features)->append_feature_obj(first_elem);
		((CCombinedDotFeatures*) test_features)->list_feature_objs();
		SG_UNREF(first_elem);
	}

	bool result=((CCombinedDotFeatures*) test_features)->append_feature_obj(f);
	if (result)
		((CCombinedDotFeatures*) test_features)->list_feature_objs();
	else
		SG_ERROR("Appending feature object failed.\n")
}

void CGUIFeatures::add_test_features(CFeatures* f)
{
	ASSERT(f)
	invalidate_test();

	if (!test_features)
	{
		test_features=new CCombinedFeatures();
		SG_REF(test_features);
	}

	if (test_features->get_feature_class()!=C_COMBINED)
	{
		CFeatures* first_elem=test_features;
		test_features=new CCombinedFeatures();
		SG_REF(test_features);
		((CCombinedFeatures*) test_features)->append_feature_obj(first_elem);
		((CCombinedFeatures*) test_features)->list_feature_objs();
		SG_UNREF(first_elem);
	}

	bool result=((CCombinedFeatures*) test_features)->append_feature_obj(f);
	if (result)
		((CCombinedFeatures*) test_features)->list_feature_objs();
	else
		SG_ERROR("Appending feature object failed.\n")
}

bool CGUIFeatures::del_last_feature_obj(char* target)
{
	CCombinedFeatures* cf=NULL;
	if (strncmp(target, "TRAIN", 5)==0)
	{
		if (!train_features)
			SG_ERROR("No train features available.\n")
		if (train_features->get_feature_class()!=C_COMBINED)
			SG_ERROR("Train features are not combined features.\n")

		cf=(CCombinedFeatures*) train_features;
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		if (!test_features)
			SG_ERROR("No test features available.\n")
		if (test_features->get_feature_class()!=C_COMBINED)
			SG_ERROR("Test features are not combined features.\n")

		cf=(CCombinedFeatures*) test_features;
	}
	else
		SG_ERROR("Unknown target %s, neither TRAIN nor TEST.\n", target)

	if (!cf->delete_feature_obj(cf->get_num_feature_obj()-1))
		SG_ERROR("No features available to delete.\n")

	return false;
}
