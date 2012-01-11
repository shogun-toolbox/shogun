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



void test_mapping_1()
{
	ParameterMap* map=new ParameterMap();

	map->put(
			new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_FLOAT64, 2),
			new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT32, 1)
	);

	map->put(
			new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT32, 1),
			new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_FLOAT64, 0)
	);

	map->put(
			new SGParamInfo("number_2", CT_SCALAR, ST_NONE, PT_INT32, 1),
			new SGParamInfo("number_to_keep", CT_SCALAR, ST_NONE, PT_INT32, 0)
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
	dummies.append_element(new SGParamInfo("number_2", CT_SCALAR, ST_NONE,
			PT_INT32, 1));

	for (index_t i=0; i<dummies.get_num_elements(); ++i)
	{
		SGParamInfo* current=dummies.get_element(i);

		char* s=current->to_string();
		SG_SPRINT("searching for: %s\n", s);
		SG_FREE(s);

		const SGParamInfo* result=map->get(current);
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
}

void print_value(const SGParamInfo* key, ParameterMap* map)
{
	const SGParamInfo* current=map->get(key);
	key->print_param_info();
	SG_SPRINT("value: ");

	if (current)
		current->print_param_info();
	else
		SG_SPRINT("no element\n");

	SG_SPRINT("\n");
}

void test_mapping_2()
{
	ParameterMap* map=new ParameterMap();

	EContainerType cfrom=CT_SCALAR;
	EContainerType cto=CT_MATRIX;

	EStructType sfrom=ST_NONE;
	EStructType sto=ST_STRING;

	EPrimitiveType pfrom=PT_BOOL;
	EPrimitiveType pto=PT_SGOBJECT;

	map->put(new SGParamInfo("1", cfrom, sfrom, pfrom, 2),
			new SGParamInfo("eins", cto, sto, pto, 1));
	map->put(new SGParamInfo("2", cfrom, sfrom, pfrom, 2),
			new SGParamInfo("zwei", cto, sto, pto, 1));
	map->put(new SGParamInfo("3", cfrom, sfrom, pfrom, 4),
			new SGParamInfo("drei", cto, sto, pto, 3));
	map->put(new SGParamInfo("4", cfrom, sfrom, pfrom, 4),
			new SGParamInfo("vier", cto, sto, pto, 3));

	SG_SPRINT("before finalization:\n");
	map->print_map();
	map->finalize_map();

	SG_SPRINT("\n\nafter finalization:\n");
	map->print_map();

	const SGParamInfo* key;

	SG_SPRINT("\n\ntesting map\n");
	key=new SGParamInfo("1", cfrom, sfrom, pfrom, 1);
	print_value(key, map);
	delete key;

	key=new SGParamInfo("2", cfrom, sfrom, pfrom, 2);
	print_value(key, map);
	delete key;

	key=new SGParamInfo("2", cto, sfrom, pfrom, 2);
	print_value(key, map);
	delete key;

	key=new SGParamInfo("2", cfrom, sto, pfrom, 2);
	print_value(key, map);
	delete key;

	key=new SGParamInfo("2", cfrom, sfrom, pto, 2);
	print_value(key, map);
	delete key;

	key=new SGParamInfo("5", cfrom, sfrom, pfrom, 4);
	print_value(key, map);
	delete key;

	delete map;
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_mapping_1();
	test_mapping_2();

	exit_shogun();

	return 0;
}

