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

#include "guilib/GUIFeatures.h"
#include "interface/SGInterface.h"


CGUIFeatures::CGUIFeatures(CSGInterface* ui_)
: CSGObject(), ui(ui_), train_features(NULL), test_features(NULL),
	ref_features(NULL)
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
	CKernel *k = ui->ui_kernel->get_kernel();
	if (k)
		k->remove_lhs();
}

void CGUIFeatures::invalidate_test()
{
	CKernel *k = ui->ui_kernel->get_kernel();
	if (k)
		k->remove_rhs() ;
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
		SG_ERROR("Unknown target %s, neither TRAIN nor TEST.\n", target);

	delete (*f_ptr);
	*f_ptr=NULL;

	if (strncmp(fclass, "SIMPLE", 6)==0)
	{
		if (strncmp(type, "REAL", 4)==0)
		{
			*f_ptr=new CRealFeatures(filename);
			//SG_DEBUG( "opening file...\n");
			//*f_ptr=new CRealFileFeatures(size, filename);

			//if (comp_features)
			//((CRealFileFeatures*) *f_ptr)->load_feature_matrix() ;
		}
		else if (strncmp(type, "BYTE", 4)==0)
		{
			///FIXME make CHAR type configurable... it is DNA by default
			*f_ptr=new CByteFeatures(DNA, filename);
		}
		else if (strncmp(type, "CHAR", 4)==0)
		{
			///FIXME make CHAR type configurable... it is DNA by default
			*f_ptr=new CCharFeatures(DNA, filename);
		}
		else if (strncmp(type, "SHORT", 5)==0)
		{
			*f_ptr=new CShortFeatures(filename);
		}
		else
		{
			SG_ERROR("Unknown type.\n");
			return false;
		}
	}
	else if (strncmp(fclass, "SPARSE", 6)==0)
	{
		io.not_implemented();
	}
	else if (strncmp(fclass, "STRING", 6)==0)
	{
		if (strncmp(type, "REAL", 4)==0)
		{
			*f_ptr=new CStringFeatures<float64_t>(filename);
		}
		else if (strncmp(type, "BYTE", 4)==0)
		{
			///FIXME make CHAR type configurable... it is DNA by default
			*f_ptr=new CStringFeatures<uint8_t>(filename, DNA);
		}
		else if (strncmp(type, "CHAR", 4)==0)
		{
			///FIXME make CHAR type configurable... it is DNA by default
			*f_ptr=new CStringFeatures<char>(filename, DNA);
		}
		else if (strncmp(type, "SHORT", 5)==0)
		{
			*f_ptr=new CStringFeatures<int16_t>(filename);
		}
		else if (strncmp(type, "WORD", 4)==0)
		{
			*f_ptr=new CStringFeatures<uint16_t>(filename);
		}
		else if (strncmp(type, "ULONG", 5)==0)
		{
			*f_ptr=new CStringFeatures<uint64_t>(filename);
		}
		else
		{
			SG_ERROR("Unknown type.\n");
			return false;
		}
	}

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
		SG_ERROR("Unknown target %s, neither TRAIN nor TEST.\n", target);

	if (*f_ptr)
	{
		if (strncmp(type, "REAL", 4)==0)
		{
			result= ((CRealFeatures*) (*f_ptr))->save(filename);
		}
		else if (strncmp(type, "BYTE", 4)==0)
		{
			result= ((CByteFeatures*) (*f_ptr))->save(filename);
		}
		else if (strncmp(type, "CHAR", 4)==0)
		{
			result= ((CCharFeatures*) (*f_ptr))->save(filename);
		}
		else if (strncmp(type, "SHORT", 5)==0)
		{
			result= ((CShortFeatures*) (*f_ptr))->save(filename);
		}
		else if (strncmp(type, "WORD", 4)==0)
		{
			result= ((CWordFeatures*) (*f_ptr))->save(filename);
		}
		else
		{
			SG_ERROR("Unknown type.\n");
			return false;
		}

		if (!result)
			SG_ERROR("Writing to file %s failed!\n", filename);
		else
		{
			SG_INFO( "Successfully written features into \"%s\" !\n", filename);
			result=true;
		}

	} else
		SG_ERROR("Set features first.\n");

	return result;
}

bool CGUIFeatures::clean(char* target)
{
	if (strncmp(target, "TRAIN", 5)==0)
		set_train_features(NULL);
	else if (strncmp(target, "TEST", 4)==0)
		set_test_features(NULL);
	else
		SG_ERROR("Unknown target %s, neither TRAIN nor TEST.\n", target);

	return true;
}

bool CGUIFeatures::obtain_by_sliding_window(
	char* target, int32_t winsize, int32_t shift, int32_t skip)
{
	ASSERT(winsize>0);
	ASSERT(shift>0);

	CFeatures* features=NULL;

	if (strncmp(target, "TRAIN", 5)==0)
	{
		invalidate_train();
		features=train_features;
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		invalidate_test();
		features=test_features;
	}
	else
	{
		SG_ERROR("Invalid target %s.\n", target);
		return false;
	}

	if (((CFeatures*) features)->get_feature_class()==C_COMBINED)
		features=((CCombinedFeatures*) features)->get_last_feature_obj();

	ASSERT(features);
	ASSERT(((CFeatures*) features)->get_feature_class()==C_STRING);

	switch (features->get_feature_type())
	{
		case F_CHAR:
			return ( ((CStringFeatures<char>*) features)->obtain_by_sliding_window(winsize, shift, skip)>0);
		case F_BYTE:
			return ( ((CStringFeatures<uint8_t>*) features)->obtain_by_sliding_window(winsize, shift, skip)>0);
		case F_WORD:
			return ( ((CStringFeatures<uint16_t>*) features)->obtain_by_sliding_window(winsize, shift, skip)>0);
		case F_ULONG:
			return ( ((CStringFeatures<uint64_t>*) features)->obtain_by_sliding_window(winsize, shift, skip)>0);
		default:
			SG_SERROR("Unsupported string features type.\n");
			return false;
	}

	return false;
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
		SG_ERROR("Invalid target %s\n", target);
		return false;
	}

	bool result=false;
	if (f_ptr)
	{
		SG_INFO( "reshape data to %d x %d\n", num_feat, num_vec);
		result=(*f_ptr)->reshape(num_feat, num_vec);

		if (!result)
			SG_ERROR("Reshaping failed.\n");
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
		combined->delete_feature_obj();
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
	CRealFeatures* src)
{
	if (src &&
		src->get_feature_class()==C_SIMPLE &&
		src->get_feature_type()==F_DREAL)
	{
		//create sparse features with 0 cache
		SG_INFO("Attempting to convert dense feature matrix to a sparse one.\n");
		CSparseFeatures<float64_t>* target=new CSparseFeatures<float64_t>(0);
		int32_t num_f=0;
		int32_t num_v=0;
		float64_t* feats=src->get_feature_matrix(num_f, num_v);
		if (target->set_full_feature_matrix(feats, num_f, num_v))
			return target;

		delete target;
	}
	else
		SG_ERROR("No SIMPLE DREAL features available.\n");

	return NULL;
}

CStringFeatures<char>* CGUIFeatures::convert_simple_char_to_string_char(
	CCharFeatures* src)
{
	if (src && src->get_feature_class()==C_SIMPLE)
	{
		int32_t num_vec=src->get_num_vectors();
		T_STRING<char>* strings=new T_STRING<char>[num_vec];
		int32_t max_len=-1;

		for (int32_t i=0; i<num_vec; i++)
		{
			bool to_free=false;
			int32_t len=0;
			char* str=src->get_feature_vector(i, len, to_free);
			strings[i].length=len ;
			for (int32_t j=0; j<len; j++)
				if (str[j]==0)
				{
					strings[i].length=j ;
					break ;
				} ;
			strings[i].string=new char[strings[i].length];

			for (int32_t j=0; j<strings[i].length; j++)
				strings[i].string[j]=str[j];

			if (strings[i].length> max_len)
				max_len=strings[i].length;

			src->free_feature_vector(str, i, to_free);
		}

		CStringFeatures<char>* target=new CStringFeatures<char>(
			new CAlphabet(src->get_alphabet()));
		target->set_features(strings, num_vec, max_len);
		return target;
	}
	else
		SG_ERROR("No features of class/type SIMPLE/CHAR available.\n");

	return NULL;
}

CRealFeatures* CGUIFeatures::convert_simple_word_to_simple_salzberg(
	CWordFeatures* src)
{
	CPluginEstimate* pie=ui->ui_pluginestimate->get_estimator();

	if (src &&
		src->get_feature_type()==F_WORD &&
		src->get_feature_class()==C_SIMPLE &&
		pie)
	{
		CRealFeatures* target=new CRealFeatures(0);
		int32_t num_feat=src->get_num_features();
		int32_t num_vec=src->get_num_vectors();
		float64_t* fm=new float64_t[num_vec*num_feat];

		if (fm)
		{
			for (int32_t i=0; i<num_vec; i++)
			{
				int32_t len=0;
				bool to_free=false;
				uint16_t* vec = src->get_feature_vector(i, len, to_free);
				ASSERT(num_feat==len);

				for (int32_t j=0; j<num_feat; j++)
					fm[i*num_feat+j]=
						pie->get_parameterwise_log_odds(vec[j], j);

				src->free_feature_vector(vec, i, to_free);
			}
			target->set_feature_matrix(fm, num_feat, num_vec);

		}
		return target;
	}
	else
		SG_ERROR("No SIMPLE WORD features or PluginEstimator available.\n");

	return NULL;
}


CTOPFeatures* CGUIFeatures::convert_string_word_to_simple_top(
	CStringFeatures<uint16_t>* src)
{
	CTOPFeatures* tf=NULL;

	if (src &&
		src->get_feature_class()==C_SIMPLE &&
		src->get_feature_type()==F_WORD)
	{
		SG_INFO("Converting to TOP features.\n");

		if (ui->ui_hmm->get_pos() && ui->ui_hmm->get_neg())
		{
			ui->ui_hmm->get_pos()->set_observations(src);
			ui->ui_hmm->get_neg()->set_observations(src);

			bool neglinear=false;
			bool poslinear=false;

			tf=new CTOPFeatures(
				0, ui->ui_hmm->get_pos(), ui->ui_hmm->get_neg(),
				neglinear, poslinear);
			ASSERT(tf->set_feature_matrix());
		}
		else
			SG_ERROR("HMMs not correctly assigned!\n");
	}
	else
		SG_ERROR("No SIMPLE WORD features available.\n");

	return tf;
}

CFKFeatures* CGUIFeatures::convert_string_word_to_simple_fk(
	CStringFeatures<uint16_t>* src)
{
	CFKFeatures* fkf=NULL;

	SG_INFO("Converting to FK features.\n");

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
			SG_ERROR("Need train features to set optimal a.\n");

		ASSERT(fkf->set_feature_matrix());

		ui->ui_hmm->get_pos()->set_observations(old_obs_pos);
		ui->ui_hmm->get_neg()->set_observations(old_obs_neg);
	}
	else
		SG_ERROR("HMMs not correctly assigned!\n");

	return fkf;
}


CRealFeatures* CGUIFeatures::convert_sparse_real_to_simple_real(
	CSparseFeatures<float64_t>* src)
{
	if (src &&
		src->get_feature_class()==C_SPARSE &&
		src->get_feature_type() == F_DREAL)
	{
		//create dense features with 0 cache
		SG_INFO("Attempting to convert sparse feature matrix to a dense one.\n");
		CRealFeatures* rf=new CRealFeatures(0);
		if (rf)
		{
			int32_t num_f=0;
			int32_t num_v=0;
			float64_t* feats=src->get_full_feature_matrix(num_f, num_v);
			rf->set_feature_matrix(feats, num_f, num_v);
			return rf;
		}
	}
	else
		SG_ERROR("No SPARSE REAL features available.\n");

	return NULL;
}

CWordFeatures* CGUIFeatures::convert_simple_char_to_simple_word(
	CCharFeatures* src, int32_t order, int32_t start, int32_t gap)
{
	if (src &&
		src->get_feature_class()==C_SIMPLE &&
		src->get_feature_type()==F_CHAR)
	{
		//create dense features with 0 cache
		SG_INFO("Converting CHAR features to WORD ones.\n");

		CWordFeatures* wf=new CWordFeatures(0);
		if (wf)
		{
			if (wf->obtain_from_char_features(src, start, order, gap))
			{
				SG_INFO("Conversion was successful.\n");
				return wf;
			}

			delete wf;
		}
	}
	else
		SG_ERROR("No SIMPLE CHAR features available.\n");

	SG_ERROR("Conversion failed.\n");
	return NULL;
}

CShortFeatures* CGUIFeatures::convert_simple_char_to_simple_short(
	CCharFeatures* src, int32_t order, int32_t start, int32_t gap)
{
	if (src &&
		src->get_feature_class()==C_SIMPLE &&
		src->get_feature_type()==F_CHAR)
	{
		//create dense features with 0 cache
		SG_INFO("Converting CHAR features to WORD ones.\n");

		CShortFeatures* sf=new CShortFeatures(0);
		if (sf)
		{
			if (sf->obtain_from_char_features(src, start, order, gap))
			{
				SG_INFO("Conversion was successful.\n");
				return sf;
			}

			delete sf;
		}
	}
	else
		SG_ERROR("No SIMPLE CHAR features available.\n");

	SG_ERROR("Conversion failed.\n");
	return NULL;
}

CRealFeatures* CGUIFeatures::convert_simple_char_to_simple_align(
	CCharFeatures* src, float64_t gap_cost)
{
	if (src &&
		src->get_feature_class()==C_SIMPLE &&
		src->get_feature_type()==F_CHAR)
	{
		//create dense features with 0 cache
		SG_INFO("Converting CHAR features to REAL ones.\n");

		CRealFeatures* rf=new CRealFeatures(0);
		if (rf)
		{
			SG_INFO("Start aligment with gapCost=%1.2f.\n", gap_cost);
			rf->Align_char_features(
				src, (CCharFeatures*) ref_features, gap_cost);
			SG_INFO("Conversion was successful.\n");
			return rf;
		}
	}
	else
		SG_ERROR("No SIMPLE CHAR features available.\n");

	SG_ERROR("Conversion failed.\n");
	return NULL;
}

bool CGUIFeatures::set_reference_features(char* target)
{
	if (strncmp(target, "TRAIN", 5)==0)
	{
		delete ref_features;
		ref_features=train_features;
		train_features=NULL;
		invalidate_train();
		return true;
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		delete ref_features;
		ref_features=test_features;
		test_features=NULL;
		invalidate_test();
		return true;
	}

	return false;
}

void CGUIFeatures::add_train_features(CFeatures* f)
{
	ASSERT(f);
	invalidate_train();

	if (!train_features)
		train_features=new CCombinedFeatures();

	if (train_features->get_feature_class()!=C_COMBINED)
	{
		CFeatures* first_elem=train_features;
		train_features=new CCombinedFeatures();
		((CCombinedFeatures*) train_features)->append_feature_obj(first_elem);
		((CCombinedFeatures*) train_features)->list_feature_objs();
	}

	bool result=((CCombinedFeatures*) train_features)->append_feature_obj(f);
	if (result)
		((CCombinedFeatures*) train_features)->list_feature_objs();
	else
		SG_ERROR("appending feature object failed\n");
}

void CGUIFeatures::add_test_features(CFeatures* f)
{
	ASSERT(f);
	invalidate_test();

	if (!test_features)
		test_features=new CCombinedFeatures();

	if (test_features->get_feature_class()!=C_COMBINED)
	{
		CFeatures* first_elem=test_features;
		test_features=new CCombinedFeatures();
		((CCombinedFeatures*) test_features)->append_feature_obj(first_elem);
		((CCombinedFeatures*) test_features)->list_feature_objs();
	}

	bool result=((CCombinedFeatures*) test_features)->append_feature_obj(f);
	if (result)
		((CCombinedFeatures*) test_features)->list_feature_objs();
	else
		SG_ERROR("Appending feature object failed.\n");
}

bool CGUIFeatures::del_last_features(char* target)
{
	CCombinedFeatures* cf=NULL;
	if (strncmp(target, "TRAIN", 5)==0)
	{
		if (!train_features)
			SG_ERROR("No train features available.\n");
		if (train_features->get_feature_class()!=C_COMBINED)
			SG_ERROR("Train features are not combined features.\n");

		cf=(CCombinedFeatures*) train_features;
	}
	else if (strncmp(target, "TEST", 4)==0)
	{
		if (!test_features)
			SG_ERROR("No test features available.\n");
		if (test_features->get_feature_class()!=C_COMBINED)
			SG_ERROR("Test features are not combined features.\n");

		cf=(CCombinedFeatures*) test_features;
	}
	else
		SG_ERROR("Unknown target %s, neither TRAIN nor TEST.\n", target);

	CFeatures* last=cf->get_last_feature_obj();
	if (last)
		return cf->delete_feature_obj();
	else
		SG_ERROR("No features available to delete.\n");

	return false;
}


#endif
