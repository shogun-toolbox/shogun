#include "lib/Mathematics.h"
#include "base/init.h"
#include "base/Parallel.h"
#include "base/Version.h"

CParallel* sg_parallel=NULL;
CIO* sg_io=NULL;
CVersion* sg_version=NULL;
CMath* sg_math=NULL;

/// function called to print normal messages
void (*sg_print_message)(FILE* target, const char* str) = NULL;

/// function called to print warning messages
void (*sg_print_warning)(FILE* target, const char* str) = NULL;

/// function called to print error messages
void (*sg_print_error)(FILE* target, const char* str) = NULL;

/// function called to cancel things
void (*sg_cancel_computations)(bool &delayed, bool &immediately)=NULL;

void init_shogun(void (*print_message)(FILE* target, const char* str),
		void (*print_warning)(FILE* target, const char* str),
		void (*print_error)(FILE* target, const char* str),
		void (*cancel_computations)(bool &delayed, bool &immediately))
{
	if (!sg_io)
		sg_io = new CIO();
	if (!sg_parallel)
		sg_parallel=new CParallel();
	if (!sg_version)
		sg_version = new CVersion();
	if (!sg_math)
		sg_math = new CMath();

	SG_REF(sg_io);
	SG_REF(sg_parallel);
	SG_REF(sg_version);
	SG_REF(sg_math);

	sg_print_message=print_message;
	sg_print_warning=print_warning;
	sg_print_error=print_error;
	sg_cancel_computations=cancel_computations;
}

void exit_shogun()
{
	sg_print_message=NULL;
	sg_print_warning=NULL;
	sg_print_error=NULL;
	sg_cancel_computations=NULL;

	SG_UNREF(sg_math);
	SG_UNREF(sg_version);
	SG_UNREF(sg_parallel);
	SG_UNREF(sg_io);
};
