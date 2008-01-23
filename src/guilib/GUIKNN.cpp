/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "guilib/GUIKNN.h"
#include "lib/io.h"
#include "gui/GUI.h"

CGUIKNN::CGUIKNN(CGUI* g) : gui(g), knn(NULL), k(0)
{
}

CGUIKNN::~CGUIKNN()
{
}

bool CGUIKNN::new_knn(CHAR* param)
{
	knn=new CKNN(); 
	return true;
}

bool CGUIKNN::train(CHAR* param)
{
	CLabels* trainlabels=gui->guilabels.get_train_labels();
	CDistance* distance=gui->guidistance.get_distance();

	bool result=false;

	if (trainlabels)
	{
		if (distance)
		{
			param=CIO::skip_spaces(param);
			k=3;
			sscanf(param, "%d", &k);

			if (knn)
			{
				knn->set_labels(trainlabels);
				knn->set_distance(distance);
				knn->set_k(k);
				result=knn->train();
			}
			else
				SG_ERROR( "no knn classifier available\n");
		}
		else
			SG_ERROR( "no distance available\n") ;
	}
	else
		SG_ERROR( "no labels available\n") ;

	return result;
}

bool CGUIKNN::test(CHAR* param)
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
	CDistance* distance=gui->guidistance.get_distance();

	if (!knn)
	{
		SG_ERROR( "no knn classifier available\n") ;
		return false ;
	}

	if (!distance)
	{
		SG_ERROR( "no distance available\n") ;
		return false ;
	}

	if (!testlabels)
	{
		SG_ERROR( "no test labels available\n") ;
		return false ;
	}

	knn->set_labels(testlabels);
	knn->set_distance(distance);

	SG_INFO( "starting knn classifier testing\n") ;
	INT len=0;
	CLabels* outlab=knn->classify(NULL);
	DREAL* output=outlab->get_labels(len);
	INT* label= testlabels->get_int_labels(len);
	ASSERT(label);

	gui->guimath.evaluate_results(output, label, len, outputfile, rocfile);

	if (rocfile)
		fclose(rocfile);
	if ((outputfile) && (outputfile!=stdout))
		fclose(outputfile);

	delete[] output;
	delete[] label;
	return true; 
}

bool CGUIKNN::load(CHAR* param)
{
  bool result=false;
  return result;
}

bool CGUIKNN::save(CHAR* param)
{
  bool result=false;
  return result;
}
#endif
