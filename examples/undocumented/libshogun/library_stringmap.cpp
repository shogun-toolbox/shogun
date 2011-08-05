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
#include <shogun/lib/StringMap.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	CStringMap* map=new CStringMap();
	map->put("1", "eins");
	map->put("2", "zwei");
	map->put("3", "drei");
	map->put("4", "vier");

	SG_SPRINT("%s->%s\n", "1", map->get("1"));
	SG_SPRINT("%s->%s\n", "2", map->get("2"));
	SG_SPRINT("%s->%s\n", "3", map->get("3"));
	SG_SPRINT("%s->%s\n", "4", map->get("4"));
	SG_SPRINT("%s->%s\n", "5", map->get("5"));

	SG_UNREF(map);

	exit_shogun();

	return 0;
}

