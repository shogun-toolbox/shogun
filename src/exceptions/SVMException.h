#ifndef _SVM_EXCEPTION_H_
#define _SVM_EXCEPTION_H_

#include "exceptions/Exception.h"

class SVMException {
      private:
         char* val;
      public:
         SVMException(const char *fmt, ... );
   
         char* get_debug_string() {
            return val;
         }
};


#endif // _SVM_EXCEPTION_H_
