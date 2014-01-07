/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2008 Christian Gehl
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <ui/GUIDistance.h>
#include <ui/SGInterface.h>

#include <string.h>

#include <lib/config.h>
#include <io/SGIO.h>
#include <io/CSVFile.h>

#include <distance/Distance.h>
#include <distance/DenseDistance.h>
#include <distance/CanberraMetric.h>
#include <distance/ChebyshewMetric.h>
#include <distance/GeodesicMetric.h>
#include <distance/JensenMetric.h>
#include <distance/ManhattanMetric.h>
#include <distance/MinkowskiMetric.h>
#include <distance/CanberraWordDistance.h>
#include <distance/ManhattanWordDistance.h>
#include <distance/HammingWordDistance.h>
#include <distance/EuclideanDistance.h>
#include <distance/SparseEuclideanDistance.h>
#include <distance/TanimotoDistance.h>
#include <distance/ChiSquareDistance.h>
#include <distance/CosineDistance.h>
#include <distance/BrayCurtisDistance.h>

#include <features/RealFileFeatures.h>
#include <features/TOPFeatures.h>
#include <features/FKFeatures.h>
#include <features/StringFeatures.h>
#include <features/DenseFeatures.h>
#include <features/Features.h>

using namespace shogun;

CGUIDistance::CGUIDistance(CSGInterface* ui_)
: CSGObject(), ui(ui_)
{
	distance=NULL;
	initialized=false;
}

CGUIDistance::~CGUIDistance()
{
	SG_UNREF(distance);
}

CDistance* CGUIDistance::get_distance()
{
	return distance;
}

bool CGUIDistance::set_distance(CDistance* dist)
{
	if (dist)
	{
		SG_REF(dist);
		SG_UNREF(distance);
		distance=dist;
		SG_DEBUG("set new distance (%p).\n", dist)

		return true;
	}
	else
		return false;
}

bool CGUIDistance::init_distance(const char* target)
{
	SG_DEBUG("init_distance start\n.")

	if (!distance)
		SG_ERROR("No distance available.\n")

	distance->set_precompute_matrix(false);
	EFeatureClass d_fclass=distance->get_feature_class();
	EFeatureType d_ftype=distance->get_feature_type();

	if (!strncmp(target, "TRAIN", 5))
	{
		CFeatures* train=ui->ui_features->get_train_features();
		if (train)
		{
			EFeatureClass fclass=train->get_feature_class();
			EFeatureType ftype=train->get_feature_type();
			if ((d_fclass==fclass || d_fclass==C_ANY || fclass==C_ANY) &&
				(d_ftype==ftype || d_ftype==F_ANY || ftype==F_ANY))

			{
				distance->init(train, train);
				initialized=true;
			}
			else
				SG_ERROR("Distance can not process this train feature type: %d %d.\n", fclass, ftype)
		}
		else
			SG_ERROR("Assign train features first.\n")
	}
	else if (!strncmp(target, "TEST", 4))
	{
		CFeatures* train=ui->ui_features->get_train_features();
		CFeatures* test=ui->ui_features->get_test_features();
		if (test)
		{
			EFeatureClass fclass=test->get_feature_class();
			EFeatureType ftype=test->get_feature_type();
			if ((d_fclass==fclass || d_fclass==C_ANY || fclass==C_ANY) &&
				(d_ftype==ftype || d_ftype==F_ANY || ftype==F_ANY))

			{
				if (!initialized)
					SG_ERROR("Distance not initialized with training examples.\n")
				else
				{
					SG_INFO("Initialising distance with TEST DATA, train: %p test %p\n", train, test)
					// lhs -> always train_features; rhs -> always test_features
					distance->init(train, test);
				}
			}
			else
				SG_ERROR("Distance can not process this test feature type: %d %d.\n", fclass, ftype)
		}
		else
			SG_ERROR("Assign train and test features first.\n")
	}
	else
	{
		SG_NOTIMPLEMENTED
		return false;
	}

	return true;

}

bool CGUIDistance::save_distance(char* param)
{
	bool result=false;
	char filename[1024]="";

	if (distance && initialized)
	{
		if ((sscanf(param, "%s", filename))==1)
		{
			CCSVFile* file=new CCSVFile(filename);
			try
			{
				distance->save(file);
			}
			catch (...)
			{
				SG_ERROR("writing to file %s failed!\n", filename)
			}

			SG_INFO("successfully written distance to \"%s\" !\n", filename)
			result=true;
			SG_UNREF(file);
		}
		else
			SG_ERROR("see help for params\n")
	}
	else
		SG_ERROR("no distance set / distance not initialized!\n")
	return result;
}

CDistance* CGUIDistance::create_generic(EDistanceType type)
{
	CDistance* dist=NULL;

	switch (type)
	{
		case D_MANHATTAN:
			dist=new CManhattanMetric(); break;
		case D_MANHATTANWORD:
			dist=new CManhattanWordDistance(); break;
		case D_CANBERRA:
			dist=new CCanberraMetric(); break;
		case D_CANBERRAWORD:
			dist=new CCanberraWordDistance(); break;
		case D_CHEBYSHEW:
			dist=new CChebyshewMetric(); break;
		case D_GEODESIC:
			dist=new CGeodesicMetric(); break;
		case D_JENSEN:
			dist=new CJensenMetric(); break;
		case D_EUCLIDEAN:
			dist=new CEuclideanDistance(); break;
		case D_SPARSEEUCLIDEAN:
			dist=new CSparseEuclideanDistance(); break;
		case D_CHISQUARE:
			dist=new CChiSquareDistance(); break;
		case D_TANIMOTO:
			dist=new CTanimotoDistance(); break;
		case D_COSINE:
			dist=new CCosineDistance(); break;
		case D_BRAYCURTIS:
			dist=new CBrayCurtisDistance(); break;
		default:
			SG_ERROR("Unknown metric/distance type %d given to create generic distance/metric.\n", type)
	}

	if (dist)
		SG_INFO("Metric/Distance of type %d created (%p).\n", type, dist)
	else
		SG_ERROR("Failed creating metric of type %d.\n", type)

	return dist;
}

CDistance* CGUIDistance::create_minkowski(float64_t k)
{
	CDistance* dist=new CMinkowskiMetric(k);
	if (dist)
		SG_INFO("Minkowski Metric created (%p), k %f.\n", dist, k)
	else
		SG_ERROR("Failed Creating Minkowski Metric, k %f.\n", k)

	return dist;
}

CDistance* CGUIDistance::create_hammingword(bool use_sign)
{
	CDistance* dist=new CHammingWordDistance(use_sign);
	if (dist)
	{
		SG_INFO("HammingWord distance created (%p), use sign %d.\n",
			dist, use_sign);
	}
	else
		SG_ERROR("Failed Creating HammingWord distance, use sign %d.\n", use_sign)

	return dist;
}
