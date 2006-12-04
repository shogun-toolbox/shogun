#ifndef _SHOGUN_EXCEPTION_H_
#define _SHOGUN_EXCEPTION_H_

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
      
//class Exception {
//      private:
//         char* val;
//      public:
//         Exception(const char *fmt, ... );
//   
//         char* get_debug_string() {
//            return val;
//         }
//};
//
//Exception::Exception(const char *fmt, ... )
//{
//   val = new char[4096];
//   char str[4096];
//   va_list list;
//   va_start(list,fmt);
//   vsnprintf(str, sizeof(str), fmt, list);
//   va_end(list);
//   strcpy(val,str);
//}

#endif // _SHOGUN_EXCEPTION_H_
