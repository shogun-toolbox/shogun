/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2007 Christian Gehl
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef HAVE_SWIG

#include "gui/GUI.h"
#include "guilib/GUIDistance.h"

#include "distance/Distance.h"
#include "distance/SimpleDistance.h"

#include "distance/Canberra.h"
#include "distance/Chebyshew.h"
#include "distance/Geodesic.h"
#include "distance/Jensen.h"
#include "distance/Manhattan.h"
#include "distance/Minkowski.h"

#include "distance/CanberraWordDistance.h"
#include "distance/ManhattanWordDistance.h"
#include "distance/HammingWordDistance.h"

#include "features/RealFileFeatures.h"
#include "features/TOPFeatures.h"
#include "features/FKFeatures.h"
#include "features/CharFeatures.h"
#include "features/StringFeatures.h"
#include "features/ByteFeatures.h"
#include "features/ShortFeatures.h"
#include "features/RealFeatures.h"
#include "features/Features.h"

#include "lib/io.h"

#include <string.h>

CGUIDistance::CGUIDistance(CGUI * gui_): CSGObject(), gui(gui_)
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
		SG_ERROR( "distance creation failed.\n");
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
				SG_ERROR( "reading from file %s failed!\n", filename);
			else
			{
				SG_INFO( "successfully read distance init data from \"%s\" !\n", filename);
				initialized=true;
				result=true;
			}

			if (file)
				fclose(file);
		}
		else
			SG_ERROR( "see help for params\n");
	}
	else
		SG_ERROR( "no kernel set!\n");
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
				SG_ERROR( "fname: %s\n", filename);
			if ((!file) || (!distance->save_init(file)))
				SG_ERROR( "writing to file %s failed!\n", filename);
			else
			{
				SG_INFO( "successfully written distance init data into \"%s\" !\n", filename);
				result=true;
			}

			if (file)
				fclose(file);
		}
		else
			SG_ERROR( "see help for params\n");
	}
	else
		SG_ERROR( "no kernel set!\n");
	return result;
}


bool CGUIDistance::init_distance(CHAR* param)
{
	CHAR target[1024]="";
	bool do_init=false;

SG_INFO( "CGUIDistance::init_distance start");
	
	if (!distance)
	{
		SG_ERROR( "no distance available\n") ;
		return false ;
	} ;

SG_INFO( "CGUIDistance::init_distance before set_precompute");
	distance->set_precompute_matrix(false);
SG_INFO( "CGUIDistance::init_distance after set_precompute");

	if ((sscanf(param, "%s", target))==1)
	{
SG_INFO( "CGUIDistance::init_distance 1 if");
		if (!strncmp(target, "TRAIN", 5))
		{
SG_INFO( "CGUIDistance::init_distance 2 if");
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
					distance->init(gui->guifeatures.get_train_features(), gui->guifeatures.get_train_features());
					initialized=true;
				}
				else
				{
					SG_ERROR( "distance can not process this feature type\n");
					return false ;
				}
			}
			else
				SG_ERROR( "assign train features first\n");
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
						SG_ERROR( "distance not initialized for training examples\n") ;
						return false ;
					}
					else
					{
						SG_INFO( "initialising distance with TEST DATA, train: %d test %d\n",gui->guifeatures.get_train_features(), gui->guifeatures.get_test_features() );
						// lhs -> always train_features; rhs -> always test_features
						distance->init(gui->guifeatures.get_train_features(), gui->guifeatures.get_test_features());

					} ;
				}
				else
				{
					SG_ERROR( "distance can not process this feature type\n");
					return false ;
				}
			}
			else
				SG_ERROR( "assign train and test features first\n");

		}
		else
			io.not_implemented();
	}
	else 
	{
		SG_ERROR( "see help for params\n");
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
				SG_ERROR( "writing to file %s failed!\n", filename);
			else
			{
				SG_INFO( "successfully written distance to \"%s\" !\n", filename);
				result=true;
			}
		}
		else
			SG_ERROR( "see help for params\n");
	}
	else
		SG_ERROR( "no distance set / distance not initialized!\n");
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
						SG_INFO( "Minkowski-Distance created\n");
					return d;
				}
				else
					SG_ERROR( "processing expects 'string string floating-point number' for Minkowski-Distance \n") ;

			}
			else
				SG_ERROR( "Minkowski-Distance expects REAL as data type \n") ;
		}
		else if (strcmp(dist_type,"MANHATTEN")==0)
		{	
			if (strcmp(data_type,"REAL")==0)
			{
				delete d;
				d= new CManhattanMetric();
				if (d)
					SG_INFO( "Manhattan-Distance created\n");
				return d;
			}
			else if (strcmp(data_type,"WORD")==0)
			{
				
				delete d;
				d=new CManhattanWordDistance();

				if (d)
				{
					SG_INFO( "ManhattenWordDistance created\n");
					return d;
				}
			}
			else
				SG_ERROR( "Manhattan-Distance expects REAL or WORD as data type \n") ;
		}
		else if (strcmp(dist_type,"HAMMING")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				INT use_sign = 0 ;
				
				
				sscanf(param, "%s %s %d", dist_type, data_type, &use_sign);
				delete d;
				d=new CHammingWordDistance(use_sign==1);

				if (d)
				{
					if (use_sign)
						SG_INFO( "HammingWordDistance with sign(count) created\n");
					else
						SG_INFO( "HammingWordDistance with count created\n");
					return d;
				}
			}
		}
		else if (strcmp(dist_type,"CANBERRA")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				delete d;
				d= new CCanberraMetric();
				if (d)
					SG_INFO( "CANBERRA-Distance created\n");
				return d;
			}
			else if (strcmp(data_type,"WORD")==0)
			{
				delete d;
				d=new CCanberraWordDistance();
				if(d)
				{
					SG_INFO("CanberraWordDistance created");
					return d;
				}
			}
			else
				SG_ERROR( "Canberra-Distance expects REAL as data type \n") ;

		}
		else if (strcmp(dist_type,"CHEBYSHEW")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				delete d;
				d= new CChebyshewMetric();
				if (d)
					SG_INFO( "Chebyshew-Distance created\n");
				return d;
			}
			else
				SG_ERROR( "Chebyshew-Distance expects REAL as data type \n") ;

		} 
		else if (strcmp(dist_type,"GEODESIC")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				delete d;
				d= new CGeodesicMetric();
				if (d)
					SG_INFO( "Geodesic-Distance created\n");
				return d;
			}
			else
				SG_ERROR( "Geodesic-Distance expects REAL as data type \n") ;

		}
		else if (strcmp(dist_type,"JENSEN")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				delete d;
				d= new CJensenMetric();
				if (d)
					SG_INFO( "Jensen-Distance created\n");
				return d;
			}
			else
				SG_ERROR( "Jense-Distance expects REAL as data type \n") ;

		}
		else
			SG_ERROR( "in this format only CANBERRA, CHEBYSHEW, GEODESIC, JENSEN, MANHATTEN, MINKOWSKI is accepted \n") ;
	} 
	else 
		SG_ERROR( "see help for params!\n");

	io.not_implemented();
	return NULL;
}

bool CGUIDistance::clean_distance(CHAR* param)
{
	delete distance;
	distance = NULL;
	return true;
}
#endif
