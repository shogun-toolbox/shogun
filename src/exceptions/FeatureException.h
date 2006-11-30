#ifndef _FEATURE_EXCEPTION_H_
#define _FEATURE_EXCEPTION_H_

#include "exceptions/Exception.h"

class FeatureException{
      private:
         char *mes;
      public:
         FeatureException(const char *_mes) {
            mes = new char[strlen(_mes)];
            strcpy(mes,_mes);
         }

         char* get_debug_string() {
            return mes;
         }
};

#endif // _FEATURE_EXCEPTION_H_
