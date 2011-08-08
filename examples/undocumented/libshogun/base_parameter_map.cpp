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

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	CParameterMap* map=new CParameterMap();
	map->put(new CSGParamInfo("1"), new CSGParamInfo("eins"));
	map->put(new CSGParamInfo("2"), new CSGParamInfo("zwei"));
	map->put(new CSGParamInfo("3"), new CSGParamInfo("drei"));
	map->put(new CSGParamInfo("4"), new CSGParamInfo("vier"));

	CSGParamInfo* info;
	CSGParamInfo* current;

	info=new CSGParamInfo("1");
	current=map->get(info);
	SG_SPRINT("key: \"%s\" ", info->m_name);
	if (current) current->print(); else SG_SPRINT("no element\n");
	SG_UNREF(current);
	SG_UNREF(info);

	info=new CSGParamInfo("2");
	current=map->get(info);
	SG_SPRINT("key: \"%s\" ", info->m_name);
	if (current) current->print(); else SG_SPRINT("no element\n");
	SG_UNREF(current);
	SG_UNREF(info);

	info=new CSGParamInfo("3");
	current=map->get(info);
	SG_SPRINT("key: \"%s\" ", info->m_name);
	if (current) current->print(); else SG_SPRINT("no element\n");
	SG_UNREF(current);
	SG_UNREF(info);

	info=new CSGParamInfo("4");
	current=map->get(info);
	SG_SPRINT("key: \"%s\" ", info->m_name);
	if (current) current->print(); else SG_SPRINT("no element\n");
	SG_UNREF(current);
	SG_UNREF(info);

	info=new CSGParamInfo("5");
	current=map->get(info);
	SG_SPRINT("key: \"%s\" ", info->m_name);
	if (current) current->print(); else SG_SPRINT("no element\n");
	SG_UNREF(current);
	SG_UNREF(info);

	SG_UNREF(map);

	exit_shogun();

	return 0;
}

