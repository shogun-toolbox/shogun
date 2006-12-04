#ifndef _ALPHABET_EXCEPTION_H_
#define _ALPHABET_EXCEPTION_H_

#include "exceptions/Exception.h"

class AlphabetException {
      private:
         char* val;
      public:
         AlphabetException(const char *fmt, ... );
   
         char* get_debug_string() {
            return val;
         }
};

#endif // _ALPHABET_EXCEPTION_H_

