#include "lib/ShogunException.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

ShogunException::ShogunException(const char* str)
{
   val = new char[4096];
   if (val)
       strncpy(val,str,4096);
   else
   {
       fprintf(stderr, "Could not even allocate memory for exception - dying.\n");
       exit(1);
   }
}
