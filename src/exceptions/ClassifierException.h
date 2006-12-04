#ifndef _CLASSIFIER_EXCEPTION_H_
#define _CLASSIFIER_EXCEPTION_H_

#include "exceptions/Exception.h"

class ClassifierException {
      private:
         char* val;
      public:
         ClassifierException(const char *fmt, ... );
   
         char* get_debug_string() {
            return val;
         }
};

#endif // _CLASSIFIER_EXCEPTION_H_

