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
#include <shogun/base/ParameterMap.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void print_value(SGParamInfo* key, ParameterMap* map)
{
	SGParamInfo* current=map->get(key);
	key->print_param_info();
	SG_SPRINT("value: ");

	if (current)
		current->print_param_info();
	else
		SG_SPRINT("no element\n");

	SG_SPRINT("\n");
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	ParameterMap* map=new ParameterMap();

	EContainerType cfrom=CT_SCALAR;
	EContainerType cto=CT_MATRIX;

	EStructType sfrom=ST_NONE;
	EStructType sto=ST_STRING;

	EPrimitiveType pfrom=PT_BOOL;
	EPrimitiveType pto=PT_SGOBJECT;

	map->put(new SGParamInfo("2", cfrom, sfrom, pfrom),
			new SGParamInfo("zwei", cto, sto, pto));
	map->put(new SGParamInfo("1", cfrom, sfrom, pfrom),
			new SGParamInfo("eins", cto, sto, pto));
	map->put(new SGParamInfo("4", cfrom, sfrom, pfrom),
			new SGParamInfo("vier", cto, sto, pto));
	map->put(new SGParamInfo("3", cfrom, sfrom, pfrom),
			new SGParamInfo("drei", cto, sto, pto));

	SG_SPRINT("before finalization:\n");
	map->print_map();
	map->finalize_map();

	SG_SPRINT("\n\nafter finalization:\n");
	map->print_map();

	SGParamInfo* key;

	SG_SPRINT("\n\ntesting map\n");
	key=new SGParamInfo("1", cfrom, sfrom, pfrom);
	print_value(key, map);
	delete key;

	key=new SGParamInfo("2", cfrom, sfrom, pfrom);
	print_value(key, map);
	delete key;

	key=new SGParamInfo("2", cto, sfrom, pfrom);
	print_value(key, map);
	delete key;

	key=new SGParamInfo("2", cfrom, sto, pfrom);
	print_value(key, map);
	delete key;

	key=new SGParamInfo("2", cfrom, sfrom, pto);
	print_value(key, map);
	delete key;

	key=new SGParamInfo("5", cfrom, sfrom, pfrom);
	print_value(key, map);
	delete key;

	delete map;

	exit_shogun();

	return 0;
}

