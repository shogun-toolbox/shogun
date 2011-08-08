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

void print_value(CSGParamInfo* key, CParameterMap* map)
{
	CSGParamInfo* current=map->get(key);
	SG_SPRINT("key: ");
	key->print();
	SG_SPRINT("value: ");

	if (current)
		current->print();
	else
		SG_SPRINT("no element\n");

	SG_SPRINT("\n");

	SG_UNREF(current);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	CParameterMap* map=new CParameterMap();

	EContainerType cfrom=CT_SCALAR;
	EContainerType cto=CT_MATRIX;

	EStructType sfrom=ST_NONE;
	EStructType sto=ST_STRING;

	EPrimitiveType pfrom=PT_BOOL;
	EPrimitiveType pto=PT_SGOBJECT;

	map->put(new CSGParamInfo("1", cfrom, sfrom, pfrom),
			new CSGParamInfo("eins", cto, sto, pto));
	map->put(new CSGParamInfo("2", cfrom, sfrom, pfrom),
			new CSGParamInfo("zwei", cto, sto, pto));
	map->put(new CSGParamInfo("3", cfrom, sfrom, pfrom),
			new CSGParamInfo("drei", cto, sto, pto));
	map->put(new CSGParamInfo("4", cfrom, sfrom, pfrom),
			new CSGParamInfo("vier", cto, sto, pto));

	CSGParamInfo* key;

	key=new CSGParamInfo("1", cfrom, sfrom, pfrom);
	print_value(key, map);
	SG_UNREF(key);

	key=new CSGParamInfo("2", cfrom, sfrom, pfrom);
	print_value(key, map);
	SG_UNREF(key);

	key=new CSGParamInfo("2", cto, sfrom, pfrom);
	print_value(key, map);
	SG_UNREF(key);

	key=new CSGParamInfo("2", cfrom, sto, pfrom);
	print_value(key, map);
	SG_UNREF(key);

	key=new CSGParamInfo("2", cfrom, sfrom, pto);
	print_value(key, map);
	SG_UNREF(key);

	key=new CSGParamInfo("5", cfrom, sfrom, pfrom);
	print_value(key, map);
	SG_UNREF(key);

	SG_UNREF(map);

	exit_shogun();

	return 0;
}

