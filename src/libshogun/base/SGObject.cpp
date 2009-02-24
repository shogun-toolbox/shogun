/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#include "base/SGObject.h"
#include "lib/io.h"
#include "lib/Mathematics.h"
#include "base/Parallel.h"
#include "base/init.h"
#include "base/Version.h"

#include <stdlib.h>
#include <stdio.h>

extern CParallel* sg_parallel;
extern CIO* sg_io;
extern CVersion* sg_version;
extern CMath* sg_math;

void CSGObject::set_global_objects()
{
	if (!sg_io || !sg_parallel || !sg_version)
	{
		fprintf(stderr, "call init_shogun() before using the library, dying.\n");
		exit(1);
	}

	SG_REF(sg_io);
	SG_REF(sg_parallel);
	SG_REF(sg_version);

	io=sg_io;
	parallel=sg_parallel;
	version=sg_version;
}
