/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/ui/GUILabels.h>
#include <shogun/ui/SGInterface.h>

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/CSVFile.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/RegressionLabels.h>

#include <string.h>

using namespace shogun;

CGUILabels::CGUILabels(CSGInterface* ui_)
: CSGObject(), ui(ui_), train_labels(NULL), test_labels(NULL)
{
}

CGUILabels::~CGUILabels()
{
	SG_UNREF(train_labels);
	SG_UNREF(test_labels);
}

bool CGUILabels::load(char* filename, char* target)
{
	CLabels* labels=NULL;

	if (strncmp(target, "TEST", 4)==0)
		labels=test_labels;
	else if (strncmp(target, "TRAIN", 5)==0)
		labels=train_labels;
	else
		SG_ERROR("Invalid target %s.\n", target)

	if (labels)
	{
		SG_UNREF(labels);
		CCSVFile* file=new CCSVFile(filename);
		labels=new CRegressionLabels(file);
		SGVector<float64_t> labs = ((CRegressionLabels*) labels)->get_labels();
		float64_t* lab=SGVector<float64_t>::clone_vector(labs.vector, labs.vlen);
		labels=infer_labels(lab, labs.vlen);

		if (labels)
		{
			if (strncmp(target, "TEST", 4)==0)
				set_test_labels(labels);
			else
				set_train_labels(labels);

			return true;
		}
		else
			SG_ERROR("Loading labels failed.\n")

		SG_UNREF(file);
	}

	return false;
}

bool CGUILabels::save(char* param)
{
	bool result=false;
	return result;
}

CLabels* CGUILabels::infer_labels(float64_t* lab, int32_t len)
{
	CLabels* labels=NULL;

	bool binary=true;
	bool multiclass=true;
	for (int32_t i=0; i<len; i++)
	{
		if (lab[i]!=-1 && lab[i]!=+1)
			binary=false;

		if (lab[i]<0 || lab[i]!=int(lab[i]))
			multiclass=false;

		if (binary == false && multiclass == false)
		{
			labels=new CRegressionLabels(SGVector<float64_t>(lab, len));
			break;
		}
	}

	if (multiclass)
		labels=new CMulticlassLabels(SGVector<float64_t>(lab, len));
	if (binary)
		labels=new CBinaryLabels(SGVector<float64_t>(lab, len));

	return labels;
}
