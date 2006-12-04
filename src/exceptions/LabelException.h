#ifndef _LABEL_EXCEPTION_H_
#define _LABEL_EXCEPTION_H_

#include "exceptions/Exception.h"

class LabelException {
      private:
         char* val;
      public:
         LabelException(const char *fmt, ... );
   
         char* get_debug_string() {
            return val;
         }
};

#endif // _LABEL_EXCEPTION_H_
