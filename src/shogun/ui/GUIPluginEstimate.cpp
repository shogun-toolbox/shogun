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

#include <ui/SGInterface.h>
#include <ui/GUIPluginEstimate.h>

#include <lib/config.h>
#include <io/SGIO.h>
#include <features/StringFeatures.h>

using namespace shogun;

CGUIPluginEstimate::CGUIPluginEstimate() : CSGObject()
{
	init();
}

CGUIPluginEstimate::CGUIPluginEstimate(CSGInterface* ui_)
: CSGObject()
{
	init();

	ui=ui_;
}

CGUIPluginEstimate::~CGUIPluginEstimate()
{
	SG_UNREF(estimator);
}

void CGUIPluginEstimate::init()
{
	ui=NULL;
	estimator=NULL;
	pos_pseudo=1e-10;
	neg_pseudo=1e-10;
}

bool CGUIPluginEstimate::new_estimator(float64_t pos, float64_t neg)
{
	SG_UNREF(estimator);
	estimator=new CPluginEstimate(pos, neg);
	SG_REF(estimator);

	if (!estimator)
		SG_ERROR("Could not create new plugin estimator, pos_pseudo %f, neg_pseudo %f\n", pos_pseudo, neg_pseudo)
	else
		SG_INFO("Created new plugin estimator (%p), pos_pseudo %f, neg_pseudo %f\n", estimator, pos_pseudo, neg_pseudo)

	return true;
}

bool CGUIPluginEstimate::train()
{
	CLabels* trainlabels=ui->ui_labels->get_train_labels();
	CStringFeatures<uint16_t>* trainfeatures=(CStringFeatures<uint16_t>*) ui->
		ui_features->get_train_features();
	bool result=false;

	if (!trainlabels)
		SG_ERROR("No labels available.\n")

	if (!trainfeatures)
		SG_ERROR("No features available.\n")

	ASSERT(trainfeatures->get_feature_type()==F_WORD)

	estimator->set_features(trainfeatures);
	estimator->set_labels(trainlabels);
	if (estimator)
		result=estimator->train();
	else
		SG_ERROR("No estimator available.\n")

	return result;
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

CLabels* CGUIPluginEstimate::apply()
{
	CFeatures* testfeatures=ui->ui_features->get_test_features();

	if (!estimator)
	{
		SG_ERROR("no estimator available")
		return 0;
	}

	if (!testfeatures)
	{
		SG_ERROR("no test features available")
		return 0;
	}

	estimator->set_features((CStringFeatures<uint16_t>*) testfeatures);

	return estimator->apply();
}

float64_t CGUIPluginEstimate::apply_one(int32_t idx)
{
	CFeatures* testfeatures=ui->ui_features->get_test_features();

	if (!estimator)
	{
		SG_ERROR("no estimator available")
		return 0;
	}

	if (!testfeatures)
	{
		SG_ERROR("no test features available")
		return 0;
	}

	estimator->set_features((CStringFeatures<uint16_t>*) testfeatures);

	return estimator->apply_one(idx);
}
