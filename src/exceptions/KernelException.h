#ifndef _KERNEL_EXCEPTION_H_
#define _KERNEL_EXCEPTION_H_

#include "exceptions/Exception.h"

class KernelException {
      private:
         char *mes;
      public:
         KernelException(const char *_mes) {
            mes = new char[strlen(_mes)];
            strcpy(mes,_mes);
         }

         char* get_debug_string() {
            return mes;
         }
};

#endif // _KERNEL_EXCEPTION_H_
