/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "guilib/GUIPluginEstimate.h"
#include "features/WordFeatures.h"
#include "lib/io.h"
#include "gui/GUI.h"

CGUIPluginEstimate::CGUIPluginEstimate(CGUI* g) : gui(g), estimator(NULL),pos_pseudo(1e-10), neg_pseudo(1e-10)
{
}

CGUIPluginEstimate::~CGUIPluginEstimate()
{
	delete estimator;
}

bool CGUIPluginEstimate::new_estimator(CHAR* param)
{
	delete estimator;
	estimator=new CPluginEstimate();
    return false;
}

bool CGUIPluginEstimate::train(CHAR* param)
{
	CLabels* trainlabels=gui->guilabels.get_train_labels();
	CWordFeatures* trainfeatures=(CWordFeatures*) gui->guifeatures.get_train_features();

	bool result=false;

	if (trainlabels)
	{
		if (trainfeatures)
		{
			ASSERT(trainfeatures->get_feature_type()==F_WORD);

			param=CIO::skip_spaces(param);
			sscanf(param, "%le %le", &pos_pseudo, &neg_pseudo);

			if (estimator)
				result=estimator->train(trainfeatures, trainlabels, pos_pseudo, neg_pseudo);
			else
				SG_ERROR( "no estimator available\n");
		}
		else
			SG_ERROR( "no features available\n") ;
	}
	else
		SG_ERROR( "no labels available\n") ;

	return result;
}

bool CGUIPluginEstimate::test(CHAR* param)
{
	CHAR outputname[1024];
	CHAR rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	INT numargs=-1;

	param=CIO::skip_spaces(param);

	numargs=sscanf(param, "%s %s", outputname, rocfname);

	if (numargs>=1)
	{
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
			SG_ERROR( "could not open %s\n",outputname);
			return false;
		}

		if (numargs==2) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				SG_ERROR( "could not open %s\n",rocfname);
				return false;
			}
		}
	}

	CLabels* testlabels=gui->guilabels.get_test_labels();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();

	if (!estimator)
	{
		SG_ERROR( "no estimator available\n") ;
		return false ;
	}

	if (!estimator->check_models())
	{
		SG_ERROR( "no models assigned\n") ;
		return false ;
	}

	if (!testfeatures || testfeatures->get_feature_class()!=C_SIMPLE || testfeatures->get_feature_type()!=F_WORD)
	{
		SG_ERROR( "no test features of type WORD available\n") ;
		return false ;
	}

	if (!testlabels)
	{
		SG_ERROR( "no test labels available\n") ;
		return false ;
	}

	SG_INFO( "starting estimator testing\n") ;
	estimator->set_testfeatures((CWordFeatures*) testfeatures);
	DREAL* output=estimator->test();

	INT len=0;
	INT total=	testfeatures->get_num_vectors();
	INT* label= testlabels->get_int_labels(len);

	SG_DEBUG( "out !!! %ld %ld\n", total, len);
	ASSERT(label);
	ASSERT(len==total);

	gui->guimath.evaluate_results(output, label, total, outputfile, rocfile);

	if (rocfile)
		fclose(rocfile);
	if ((outputfile) && (outputfile!=stdout))
		fclose(outputfile);

	delete[] output;
	delete[] label;
	return true;
}

bool CGUIPluginEstimate::load(CHAR* param)
{
  bool result=false;
  return result;
}

bool CGUIPluginEstimate::save(CHAR* param)
{
  bool result=false;
  return result;
}

CLabels* CGUIPluginEstimate::classify(CLabels* output)
{
	CFeatures* testfeatures=gui->guifeatures.get_test_features();

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

	estimator->set_testfeatures((CWordFeatures*) testfeatures);

	return estimator->classify(output);
}

DREAL CGUIPluginEstimate::classify_example(INT idx)
{
	CFeatures* testfeatures=gui->guifeatures.get_test_features();

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

	estimator->set_testfeatures((CWordFeatures*) testfeatures);

	return estimator->classify_example(idx);
}
#endif
