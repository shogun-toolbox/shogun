#include "exceptions/KernelException.h"

KernelException::KernelException(const char *fmt, ... )
{
   val = new char[4096];
   char str[4096];
   va_list list;
   va_start(list,fmt);
   vsnprintf(str, sizeof(str), fmt, list);
   va_end(list);
   strcpy(val,str);
}
