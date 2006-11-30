#ifndef _CLASSIFIER_EXCEPTION_H_
#define _CLASSIFIER_EXCEPTION_H_

#include "exceptions/Exception.h"

class ClassifierException {
      private:
         char *mes;
      public:
         ClassifierException(const char *_mes) {
            mes = new char[strlen(_mes)];
            strcpy(mes,_mes);
         }

         char* get_debug_string() {
            return mes;
         }
};

#endif // _CLASSIFIER_EXCEPTION_H_

