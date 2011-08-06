/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#include <shogun/classifier/vw/vw_example.h>

using namespace shogun;

VwExample::VwExample(): tag(), indices(), atomics(),
			num_features(0), pass(0),
			final_prediction(0.), loss(0),
			eta_round(0.), global_weight(0),
			example_t(0), total_sum_feat_sq(1), sorted(false)
{
	ld = new VwLabel();
}

VwExample::~VwExample()
{
	if (ld)
		delete ld;
	if (tag.end_array != tag.begin)
	{
		free(tag.begin);
		tag.end_array = tag.begin;
	}

	for (index_t j = 0; j < 256; j++)
	{
		if (atomics[j].begin != atomics[j].end_array)
			free(atomics[j].begin);
	}
	free(indices.begin);
}

void VwExample::reset_members()
{
	num_features = 0;
	total_sum_feat_sq = 1;
	example_counter = 0;
	global_weight = 0;
	example_t = 0;
	eta_round = 0;
	final_prediction = 0;
	loss = 0;

	for (index_t* i = indices.begin; i != indices.end; i++)
	{
		atomics[*i].erase();
		sum_feat_sq[*i]=0;
	}

	indices.erase();
	tag.erase();
}
