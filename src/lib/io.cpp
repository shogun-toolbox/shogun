#include "lib/io.h"
#include "lib/common.h"
#include "lib/Mathmatics.h"

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>

FILE* CIO::target=stdout;
LONG CIO::last_progress_time=0 ;
LONG CIO::progress_start_time=0 ;
REAL CIO::last_progress=1 ;

CIO::CIO()
{

}

void CIO::message(EMessageType prio, const CHAR *fmt, ... )
{
	check_target();
	print_message_prio(prio, target);
    va_list list;
    va_start(list,fmt);
    vfprintf(target,fmt,list);
    va_end(list);
    fflush(target);
}

void CIO::progress(REAL current_val, REAL min_val, REAL max_val, INT decimals, const char* prefix)
{
	LONG runtime = get_runtime() ;
	
	char str[1000];
	REAL v=-1, estimate=0, total_estimate=0 ;
	check_target();

	if (max_val-min_val)
		v=100*(current_val-min_val+1)/(max_val-min_val+1);

	if (decimals < 1)
		decimals = 1;

	//fprintf(stdout, "runtime=%ld  %f %f\n", runtime, v, last_progress) ;

	if (last_progress>v)
	{
		last_progress_time = runtime ;
		progress_start_time = runtime;
		last_progress = v ;
		//fprintf(stdout,"reset\n") ;
	}
	else
	{
		if (v>100) v=100.0 ;
		if (v<=0) v=1e-6 ;
		last_progress = v-1e-5 ; ;
		
		if ((v!=100.0) && (runtime - last_progress_time<100))
			return ;
		
		last_progress_time = runtime ;
		estimate = (1-v/100)*(last_progress_time-progress_start_time)/(v/100) ;
		total_estimate = (last_progress_time-progress_start_time)/(v/100) ;
	}
	
	if (estimate/100>120)
	{
		sprintf(str, "%%s %%%d.%df%%    %%1.1f minutes remaining    %%1.1f minutes total    \r",decimals+3, decimals);
		fprintf(target, str, prefix, v, estimate/100/60, total_estimate/100/60);
	}
	else
	{
		sprintf(str, "%%s %%%d.%df%%    %%1.1f seconds remaining    %%1.1f seconds total    \r",decimals+3, decimals);
		fprintf(target, str, prefix, v, estimate/100, total_estimate/100);
	}
	
    fflush(target);
}

void CIO::buffered_message(EMessageType prio, const CHAR *fmt, ... )
{
	check_target();
	print_message_prio(prio, target);
    va_list list;
    va_start(list,fmt);
    vfprintf(target,fmt,list);
    va_end(list);
}

CHAR* CIO::skip_spaces(CHAR* str)
{
	INT i=0;

	if (str)
	{
		for (i=0; isspace(str[i]); i++);

		return &str[i];
	}
	else 
		return str;
}

void CIO::set_target(FILE* t)
{
	target=t;
}

void CIO::check_target()
{
	if (!target)
		target=stdout;
}

void CIO::print_message_prio(EMessageType prio, FILE* target)
{
	switch (prio)
	{
		case M_DEBUG:
			fprintf(target, "[DEBUG] ");
			break;
		case M_INFO:
			//fprintf(target, "[INFO]");
			break;
		case M_NOTICE:
			fprintf(target, "[NOTICE] ");
			break;
		case M_WARN:
			fprintf(target, "[WARN] ");
			break;
		case M_ERROR:
			fprintf(target, "[ERROR] ");
			break;
		case M_CRITICAL:
			fprintf(target, "[CRITICAL] ");
			break;
		case M_ALERT:
			fprintf(target, "[ALERT] ");
			break;
		case M_EMERGENCY:
			fprintf(target, "[EMERGENCY] ");
			break;
		case M_MESSAGEONLY:
			break;
		default:
			break;
	}
}
