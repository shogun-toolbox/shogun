/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "gui/GUI.h"
#include "guilib/GUIDistance.h"

#include "distance/Distance.h"
#include "distance/RealDistance.h"

#include "distance/Canberra.h"
#include "distance/Chebyshew.h"
#include "distance/Geodesic.h"
#include "distance/Jensen.h"
#include "distance/Manhattan.h"
#include "distance/Minkowski.h"

#include "features/RealFileFeatures.h"
#include "features/TOPFeatures.h"
#include "features/FKFeatures.h"
#include "features/CharFeatures.h"
#include "features/StringFeatures.h"
#include "features/ByteFeatures.h"
#include "features/ShortFeatures.h"
#include "features/RealFeatures.h"
#include "features/SparseRealFeatures.h"
#include "features/Features.h"

#include "lib/io.h"

#include <string.h>

CGUIDistance::CGUIDistance(CGUI * gui_): gui(gui_)
{
	distance=NULL;
	initialized=false;
}

CGUIDistance::~CGUIDistance()
{
	delete distance;
}

CDistance* CGUIDistance::get_distance()
{
	return distance;
}

bool CGUIDistance::set_distance(CHAR* param)
{
	CDistance* d=create_distance(param);

	if (distance && d)
		delete distance;

	if (d)
	{
		distance=d;
		return true;
	}
	else
	{
		CIO::message(M_ERROR, "distance creation failed.\n");
		return false;
	}
}

bool CGUIDistance::load_distance_init(CHAR* param)
{
	bool result=false;
	CHAR filename[1024]="";

	if (distance)
	{
		if ((sscanf(param, "%s", filename))==1)
		{
			FILE* file=fopen(filename, "r");
			if ((!file) || (!distance->load_init(file)))
				CIO::message(M_ERROR, "reading from file %s failed!\n", filename);
			else
			{
				CIO::message(M_INFO, "successfully read distance init data from \"%s\" !\n", filename);
				initialized=true;
				result=true;
			}

			if (file)
				fclose(file);
		}
		else
			CIO::message(M_ERROR, "see help for params\n");
	}
	else
		CIO::message(M_ERROR, "no kernel set!\n");
	return result;
}

bool CGUIDistance::save_distance_init(CHAR* param)
{
	bool result=false;
	CHAR filename[1024]="";

	if (distance)
	{
		if ((sscanf(param, "%s", filename))==1)
		{
			FILE* file=fopen(filename, "w");
			if (!file)
				CIO::message(M_ERROR, "fname: %s\n", filename);
			if ((!file) || (!distance->save_init(file)))
				CIO::message(M_ERROR, "writing to file %s failed!\n", filename);
			else
			{
				CIO::message(M_INFO, "successfully written distance init data into \"%s\" !\n", filename);
				result=true;
			}

			if (file)
				fclose(file);
		}
		else
			CIO::message(M_ERROR, "see help for params\n");
	}
	else
		CIO::message(M_ERROR, "no kernel set!\n");
	return result;
}


bool CGUIDistance::init_distance(CHAR* param)
{
	CHAR target[1024]="";
	bool do_init=false;

CIO::message(M_INFO, "CGUIDistance::init_distance start");
	
	if (!distance)
	{
		CIO::message(M_ERROR, "no distance available\n") ;
		return false ;
	} ;

CIO::message(M_INFO, "CGUIDistance::init_distance before set_precompute");
	distance->set_precompute_matrix(false);
CIO::message(M_INFO, "CGUIDistance::init_distance after set_precompute");

	if ((sscanf(param, "%s", target))==1)
	{
CIO::message(M_INFO, "CGUIDistance::init_distance 1 if");
		if (!strncmp(target, "TRAIN", 5))
		{
CIO::message(M_INFO, "CGUIDistance::init_distance 2 if");
			do_init=true;
			if (gui->guifeatures.get_train_features())
			{
				if ( (distance->get_feature_class() == gui->guifeatures.get_train_features()->get_feature_class() 
							|| gui->guifeatures.get_train_features()->get_feature_class() == C_ANY 
							|| distance->get_feature_class() == C_ANY ) &&
						(distance->get_feature_type() == gui->guifeatures.get_train_features()->get_feature_type() 
						 || gui->guifeatures.get_train_features()->get_feature_type() == F_ANY 
						 || distance->get_feature_type() == F_ANY) )
				{
					distance->init(gui->guifeatures.get_train_features(), gui->guifeatures.get_train_features(), do_init);
					initialized=true;
				}
				else
				{
					CIO::message(M_ERROR, "distance can not process this feature type\n");
					return false ;
				}
			}
			else
				CIO::message(M_ERROR, "assign train features first\n");
		}
		else if (!strncmp(target, "TEST", 5))
		{
			if (gui->guifeatures.get_train_features() && gui->guifeatures.get_test_features())
			{
				if ( (distance->get_feature_class() == gui->guifeatures.get_train_features()->get_feature_class() 
							|| gui->guifeatures.get_train_features()->get_feature_class() == C_ANY 
							|| distance->get_feature_class() == C_ANY ) &&
						(distance->get_feature_class() == gui->guifeatures.get_test_features()->get_feature_class() 
							|| gui->guifeatures.get_test_features()->get_feature_class() == C_ANY 
							|| distance->get_feature_class() == C_ANY ) &&
						(distance->get_feature_type() == gui->guifeatures.get_train_features()->get_feature_type() 
						 || gui->guifeatures.get_train_features()->get_feature_type() == F_ANY 
						 || distance->get_feature_type() == F_ANY ) &&
						(distance->get_feature_type() == gui->guifeatures.get_test_features()->get_feature_type() 
						 || gui->guifeatures.get_test_features()->get_feature_type() == F_ANY 
						 || distance->get_feature_type() == F_ANY ) )
				{
					if (!initialized)
					{
						CIO::message(M_ERROR, "distance not initialized for training examples\n") ;
						return false ;
					}
					else
					{
						CIO::message(M_INFO, "initialising distance with TEST DATA, train: %d test %d\n",gui->guifeatures.get_train_features(), gui->guifeatures.get_test_features() );
						// lhs -> always train_features; rhs -> always test_features
						distance->init(gui->guifeatures.get_train_features(), gui->guifeatures.get_test_features(), do_init);						
					} ;
				}
				else
				{
					CIO::message(M_ERROR, "distance can not process this feature type\n");
					return false ;
				}
			}
			else
				CIO::message(M_ERROR, "assign train and test features first\n");

		}
		else
			CIO::not_implemented();
	}
	else 
	{
		CIO::message(M_ERROR, "see help for params\n");
		return false;
	}

	return true;
}

bool CGUIDistance::save_distance(CHAR* param)
{
	bool result=false;
	CHAR filename[1024]="";

	if (distance && initialized)
	{
		if ((sscanf(param, "%s", filename))==1)
		{
			if (!distance->save(filename))
				CIO::message(M_ERROR, "writing to file %s failed!\n", filename);
			else
			{
				CIO::message(M_INFO, "successfully written distance to \"%s\" !\n", filename);
				result=true;
			}
		}
		else
			CIO::message(M_ERROR, "see help for params\n");
	}
	else
		CIO::message(M_ERROR, "no distance set / distance not initialized!\n");
	return result;
}

CDistance* CGUIDistance::create_distance(CHAR* param)
{
	CHAR dist_type[1024]="";
	CHAR data_type[1024]="";
	param=CIO::skip_spaces(param);
	CDistance* d=NULL;
	
	//note the different args COMBINED <cachesize>
	 
	if (sscanf(param, "%s %s", dist_type, data_type) == 2)
	{
		if (strcmp(dist_type,"MINKOWSKI")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				DREAL p = 1 ;
				if(sscanf(param, "%s %s %lf", dist_type, data_type,&p)==3)
				{
					delete d;
					d= new CMinkowskiMetric(p);
					if (d)
						CIO::message(M_INFO, "Minkowski-Distance created\n");
					return d;
				}
				else
					CIO::message(M_ERROR, "processing expects 'string string floating-point number' for Minkowski-Distance \n") ;

			}
			else
				CIO::message(M_ERROR, "Minkowski-Distance expects REAL as data type \n") ;
		}
		else if (strcmp(dist_type,"MANHATTEN")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				delete d;
				d= new CManhattanMetric();
				if (d)
					CIO::message(M_INFO, "Manhattan-Distance created\n");
				return d;
			}
			else
				CIO::message(M_ERROR, "Manhattan-Distance expects REAL as data type \n") ;
		}
		else if (strcmp(dist_type,"CANBERRA")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				delete d;
				d= new CCanberraMetric();
				if (d)
					CIO::message(M_INFO, "CANBERRA-Distance created\n");
				return d;
			}
			else
				CIO::message(M_ERROR, "Canberra-Distance expects REAL as data type \n") ;

		}
		else if (strcmp(dist_type,"CHEBYSHEW")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				delete d;
				d= new CChebyshewMetric();
				if (d)
					CIO::message(M_INFO, "Chebyshew-Distance created\n");
				return d;
			}
			else
				CIO::message(M_ERROR, "Chebyshew-Distance expects REAL as data type \n") ;

		} 
		else if (strcmp(dist_type,"GEODESIC")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				delete d;
				d= new CGeodesicMetric();
				if (d)
					CIO::message(M_INFO, "Geodesic-Distance created\n");
				return d;
			}
			else
				CIO::message(M_ERROR, "Geodesic-Distance expects REAL as data type \n") ;

		}
		else if (strcmp(dist_type,"JENSEN")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				delete d;
				d= new CJensenMetric();
				if (d)
					CIO::message(M_INFO, "Jensen-Distance created\n");
				return d;
			}
			else
				CIO::message(M_ERROR, "Jense-Distance expects REAL as data type \n") ;

		}   
		else
			CIO::message(M_ERROR, "in this format only CANBERRA, CHEBYSHEW, GEODESIC, JENSEN, MANHATTEN, MINKOWSKI is accepted \n") ;
	} 
	else 
		CIO::message(M_ERROR, "see help for params!\n");

	CIO::not_implemented();
	return NULL;
}

bool CGUIDistance::clean_distance(CHAR* param)
{
	delete distance;
	distance = NULL;
	return true;
}
