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

#include "SGInterface.h"
#include "GUIPluginEstimate.h"

#include <shogun/lib/config.h>
#include <shogun/lib/io.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

CGUIPluginEstimate::CGUIPluginEstimate(CSGInterface* ui_)
: CSGObject(), ui(ui_), estimator(NULL),
	pos_pseudo(1e-10), neg_pseudo(1e-10)
{
}

CGUIPluginEstimate::~CGUIPluginEstimate()
{
	SG_UNREF(estimator);
}

bool CGUIPluginEstimate::new_estimator(float64_t pos, float64_t neg)
{
	SG_UNREF(estimator);
	estimator=new CPluginEstimate(pos, neg);
	SG_REF(estimator);

	if (!estimator)
		SG_ERROR("Could not create new plugin estimator, pos_pseudo %f, neg_pseudo %f\n", pos_pseudo, neg_pseudo);
	else
		SG_INFO("Created new plugin estimator (%p), pos_pseudo %f, neg_pseudo %f\n", estimator, pos_pseudo, neg_pseudo);

	return true;
}

bool CGUIPluginEstimate::train()
{
	CLabels* trainlabels=ui->ui_labels->get_train_labels();
	CStringFeatures<uint16_t>* trainfeatures=(CStringFeatures<uint16_t>*) ui->
		ui_features->get_train_features();
	bool result=false;

	if (!trainlabels)
		SG_ERROR("No labels available.\n");

	if (!trainfeatures)
		SG_ERROR("No features available.\n");

	ASSERT(trainfeatures->get_feature_type()==F_WORD);

	estimator->set_features(trainfeatures);
	estimator->set_labels(trainlabels);
	if (estimator)
		result=estimator->train();
	else
		SG_ERROR("No estimator available.\n");

	return result;
}

bool CGUIPluginEstimate::test(char* filename_out, char* filename_roc)
{
	FILE* file_out=stdout;
	FILE* file_roc=NULL;

	if (!estimator)
		SG_ERROR("No estimator available.\n");

	if (!estimator->check_models())
		SG_ERROR("No models assigned.\n");

	CLabels* testlabels=ui->ui_labels->get_test_labels();
	if (!testlabels)
		SG_ERROR("No test labels available.\n");

	CFeatures* testfeatures=ui->ui_features->get_test_features();
	if (!testfeatures || testfeatures->get_feature_class()!=C_SIMPLE ||
		testfeatures->get_feature_type()!=F_WORD)
		SG_ERROR("No test features of type WORD available.\n");

	if (filename_out)
	{
		file_out=fopen(filename_out, "w");

		if (!file_out)
			SG_ERROR("Could not open file %s.\n", filename_out);

		if (filename_roc)
		{
			file_roc=fopen(filename_roc, "w");
			if (!file_roc)
				SG_ERROR("Could not open ROC file %s\n", filename_roc);
		}
	}

	SG_INFO("Starting estimator testing.\n");
	estimator->set_features((CStringFeatures<uint16_t>*) testfeatures);
	int32_t len=0;
	float64_t* output=estimator->classify()->get_labels(len);

	int32_t total=testfeatures->get_num_vectors();
	int32_t* label=testlabels->get_int_labels(len);

	SG_DEBUG("out !!! %ld %ld.\n", total, len);
	ASSERT(label);
	ASSERT(len==total);

	ui->ui_math->evaluate_results(output, label, total, file_out, file_roc);

	if (file_roc)
		fclose(file_roc);
	if (file_out && file_out!=stdout)
		fclose(file_out);

	delete[] output;
	delete[] label;
	return true;
}

bool CGUIPluginEstimate::load(char* param)
{
  bool result=false;
  return result;
}

bool CGUIPluginEstimate::save(char* param)
{
  bool result=false;
  return result;
}

CLabels* CGUIPluginEstimate::classify()
{
	CFeatures* testfeatures=ui->ui_features->get_test_features();

	if (!estimator)
	{
		SG_ERROR( "no estimator available") ;
		return 0;
	}

	if (!testfeatures)
	{
		SG_ERROR( "no test features available") ;
		return 0;
	}

	estimator->set_features((CStringFeatures<uint16_t>*) testfeatures);

	return estimator->classify();
}

float64_t CGUIPluginEstimate::classify_example(int32_t idx)
{
	CFeatures* testfeatures=ui->ui_features->get_test_features();

	if (!estimator)
	{
		SG_ERROR( "no estimator available") ;
		return 0;
	}

	if (!testfeatures)
	{
		SG_ERROR( "no test features available") ;
		return 0;
	}

	estimator->set_features((CStringFeatures<uint16_t>*) testfeatures);

	return estimator->classify_example(idx);
}
