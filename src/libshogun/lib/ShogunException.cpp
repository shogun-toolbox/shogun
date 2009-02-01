#include "lib/ShogunException.h"
#include "lib/Signal.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

ShogunException::ShogunException(const char* str)
{
#ifndef WIN32
	CSignal::unset_handler();
#endif
   val = new char[4096];
   if (val)
       strncpy(val,str,4096);
   else
   {
       fprintf(stderr, "Could not even allocate memory for exception - dying.\n");
       exit(1);
   }
}
