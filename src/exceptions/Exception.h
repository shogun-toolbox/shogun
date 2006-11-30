#ifndef _SHOGUN_EXCEPTION_H_
#define _SHOGUN_EXCEPTION_H_

#include <cstring>
using namespace std;

class Exception {
      private:
         char *mes;
      public:
         Exception(const char *_mes) {
            mes = new char[strlen(_mes)];
            strcpy(mes,_mes);
         }

         char* get_debug_string() {
            return mes;
         }
};

#endif // _SHOGUN_EXCEPTION_H_
