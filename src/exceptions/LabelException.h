#ifndef _LABEL_EXCEPTION_H_
#define _LABEL_EXCEPTION_H_

#include <cstring>
using namespace std;

class LabelException{
      private:
         char *mes;
      public:
         LabelException(const char *_mes) {
            mes = new char[strlen(_mes)];
            strcpy(mes,_mes);
         }

         char* get_debug_string() {
            return mes;
         }
};

#endif // _LABEL_EXCEPTION_H_
