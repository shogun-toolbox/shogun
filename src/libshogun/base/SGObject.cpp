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
#include "base/Parallel.h"
#include "base/init.h"
#include "base/Version.h"
#include "lib/Parameter.h"

#include <stdlib.h>
#include <stdio.h>


namespace shogun
{
	class CMath;
	class Parallel;
	class IO;
	class Version;

	extern CMath* sg_math;
	extern Parallel* sg_parallel;
	extern IO* sg_io;
	extern Version* sg_version;

}

using namespace shogun;

void
CSGObject::set_global_objects(void)
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

void
CSGObject::unset_global_objects(void)
{
	SG_UNREF(version);
	SG_UNREF(parallel);
	SG_UNREF(io);
}

void CSGObject::set_io(IO* new_io)
{
	SG_UNREF(sg_io);
	sg_io=new_io;
	SG_REF(sg_io);
}

IO* CSGObject::get_io()
{
	SG_REF(sg_io);
	return sg_io;
}

void CSGObject::set_parallel(Parallel* new_parallel)
{
	SG_UNREF(sg_parallel);
	sg_parallel=new_parallel;
	SG_REF(sg_parallel);
}

Parallel* CSGObject::get_parallel()
{
	SG_REF(sg_parallel);
	return sg_parallel;
}

void CSGObject::set_version(Version* new_version)
{
	SG_UNREF(sg_version);
	sg_version=new_version;
	SG_REF(sg_version);
}

Version* CSGObject::get_version()
{
	SG_REF(sg_version);
	return sg_version;
}
