#ifndef _SHOGUN_EXCEPTION_H_
#define _SHOGUN_EXCEPTION_H_

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
      
class ShogunException {
      private:
         char* val;
      public:
         ShogunException(const char *fmt, ... );
   
         char* get_debug_string() {
            return val;
         }
};

#endif // _SHOGUN_EXCEPTION_H_
