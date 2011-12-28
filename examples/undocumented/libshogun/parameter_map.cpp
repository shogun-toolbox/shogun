/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/ParameterMap.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}


int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	ParameterMap* map=new ParameterMap();

	map->put(
			new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_FLOAT64, 2),
			new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT32, 1)
	);

	map->put(
			new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT32, 1),
			new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_FLOAT64, 0)
	);

	/* finalizing the map is needed before accessing it */
	map->finalize_map();

	map->print_map();
	SG_SPRINT("\n");


	/* get some elements from map, one/two ARE in map, three and four are NOT */
	DynArray<SGParamInfo*> dummies;
	dummies.append_element(new SGParamInfo("number", CT_SCALAR, ST_NONE,
			PT_INT32, 1));
	dummies.append_element(new SGParamInfo("number", CT_SCALAR, ST_NONE,
			PT_FLOAT64, 2));
	dummies.append_element(new SGParamInfo("number", CT_SCALAR, ST_NONE,
				PT_INT32, 2));
	dummies.append_element(new SGParamInfo("number", CT_SCALAR, ST_NONE,
			PT_FLOAT64, 0));

	for (index_t i=0; i<dummies.get_num_elements(); ++i)
	{
		SGParamInfo* current=dummies.get_element(i);

		char* s=current->to_string();
		SG_SPRINT("searching for: %s\n", s);
		SG_FREE(s);

		if (i==2)
		{

		}

		SGParamInfo* result=map->get(current);
		if (result)
		{
			s=result->to_string();
			SG_SPRINT("found: %s\n\n", s);
			SG_FREE(s);
		}
		else
			SG_SPRINT("nothing found\n\n");

		delete current;
	}


	delete map;

	exit_shogun();

	return 0;
}

