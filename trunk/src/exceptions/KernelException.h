#ifndef _KERNEL_EXCEPTION_H_
#define _KERNEL_EXCEPTION_H_

#include "exceptions/Exception.h"

class KernelException {
      private:
         char* val;
      public:
         KernelException(const char *fmt, ... );
   
         char* get_debug_string() {
            return val;
         }
};

#endif // _KERNEL_EXCEPTION_H_
