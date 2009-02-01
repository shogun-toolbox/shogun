#include "base/SGObject.h"
#include "lib/io.h"
#include "lib/Mathematics.h"
#include "base/Parallel.h"
#include "base/Version.h"

CParallel* sg_parallel=NULL;
CIO* sg_io=NULL;
CVersion* sg_version=NULL;

//this creates a math object for the purpose of the constructor to be called at least once
volatile CMath math;

extern CParallel* sg_parallel;
extern CIO* sg_io;
extern CVersion* sg_version;

void CSGObject::set_global_objects()
{
	if (!sg_io)
		sg_io = new CIO();
	if (!sg_parallel)
		sg_parallel=new CParallel();
	if (!sg_version)
		sg_version = new CVersion();

	SG_REF(sg_io);
	SG_REF(sg_parallel);
	SG_REF(sg_version);

	io=sg_io;
	parallel=sg_parallel;
	version=sg_version;
}
