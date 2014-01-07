/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <base/init.h>
#include <base/Parameter.h>
#include <base/ParameterMap.h>

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
	SG_SPRINT("\n\before finalization:\n");
	map->finalize_map();

	SG_SPRINT("\n\nafter finalization:\n");
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

		DynArray<const SGParamInfo*>* result=map->get(current);
		if (result)
		{
			for (index_t j=0; j<result->get_num_elements(); ++j)
			{
				s=result->get_element(j)->to_string();
				SG_SPRINT("found: %s\n\n", s);
				SG_FREE(s);
			}
		}
		else
			SG_SPRINT("nothing found\n\n");

		delete current;
	}

	delete map;
}

void print_value(const SGParamInfo* key, ParameterMap* map)
{
	DynArray<const SGParamInfo*>* current=map->get(key);
	key->print_param_info();
	SG_SPRINT("value: ");

	if (current)
	{
		for (index_t i=0; i<current->get_num_elements(); ++i)
			current->get_element(i)->print_param_info("\t");
	}
	else
		SG_SPRINT("no elements\n");

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

void test_mapping_0()
{
	/* test multiple values per key */
	ParameterMap* map=new ParameterMap();

	EContainerType cfrom=CT_SCALAR;
	EContainerType cto=CT_MATRIX;

	EStructType sfrom=ST_NONE;
	EStructType sto=ST_STRING;

	EPrimitiveType pfrom=PT_BOOL;
	EPrimitiveType pto=PT_SGOBJECT;

	/* 3 equal keys */
	map->put(new SGParamInfo("1", cfrom, sfrom, pfrom, 2),
			new SGParamInfo("eins a", cto, sto, pto, 1));

	map->put(new SGParamInfo("1", cfrom, sfrom, pfrom, 2),
			new SGParamInfo("eins b", cto, sto, pto, 1));

	map->put(new SGParamInfo("1", cfrom, sfrom, pfrom, 2),
			new SGParamInfo("eins c", cto, sto, pto, 1));

	/* 2 equal keys */
	map->put(new SGParamInfo("2", cfrom, sfrom, pfrom, 2),
			new SGParamInfo("zwei a", cto, sto, pto, 1));

	map->put(new SGParamInfo("2", cfrom, sfrom, pfrom, 2),
			new SGParamInfo("zwei b", cto, sto, pto, 1));

	map->finalize_map();

	SG_SPRINT("printing finalized map\n");
	map->print_map();

	/* assert that all is there */
	DynArray<const SGParamInfo*>* result;
	bool found;

	/* key 0 */
	result=map->get(SGParamInfo("1", cfrom, sfrom, pfrom, 2));
	ASSERT(result);

	/* first value element */
	found=false;
	for (index_t i=0; i<result->get_num_elements(); ++i)
	{
		if (*result->get_element(i) == SGParamInfo("eins a", cto, sto, pto, 1))
			found=true;
	}
	ASSERT(found);

	/* second value element */
	found=false;
	for (index_t i=0; i<result->get_num_elements(); ++i)
	{
		if (*result->get_element(i) == SGParamInfo("eins b", cto, sto, pto, 1))
			found=true;
	}
	ASSERT(found);

	/* third value element */
	found=false;
	for (index_t i=0; i<result->get_num_elements(); ++i)
	{
		if (*result->get_element(i) == SGParamInfo("eins c", cto, sto, pto, 1))
			found=true;
	}
	ASSERT(found);

	/* key 1 */
	result=map->get(SGParamInfo("2", cfrom, sfrom, pfrom, 2));
	ASSERT(result);

	/* first value element */
	found=false;
	for (index_t i=0; i<result->get_num_elements(); ++i)
	{
		if (*result->get_element(i) == SGParamInfo("zwei a", cto, sto, pto, 1))
			found=true;
	}
	ASSERT(found);

	/* second value element */
	found=false;
	for (index_t i=0; i<result->get_num_elements(); ++i)
	{
		if (*result->get_element(i) == SGParamInfo("zwei b", cto, sto, pto, 1))
			found=true;
	}
	ASSERT(found);

	delete map;
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_mapping_0();
	test_mapping_1();
	test_mapping_2();

	exit_shogun();

	return 0;
}
