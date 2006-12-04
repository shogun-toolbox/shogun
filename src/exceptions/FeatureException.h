#ifndef _FEATURE_EXCEPTION_H_
#define _FEATURE_EXCEPTION_H_ 

#include "exceptions/Exception.h"

class FeatureException {
      private:
         char* val;
      public:
         FeatureException(const char *fmt, ... );
   
         char* get_debug_string() {
            return val;
         }
};

#endif // _FEATURE_EXCEPTION_H_
