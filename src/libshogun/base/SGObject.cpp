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
