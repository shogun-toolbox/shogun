#include <io/SGIO.h>
#include <stdio.h>

void sg_global_print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void sg_global_print_warning(FILE* target, const char* str)
{
    fprintf(target, "%s", str);
}

void sg_global_print_error(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void sg_global_cancel_computations(bool &delayed, bool &immediately)
{
}
