#ifndef _PLUGIN_EXCEPTION_H_
#define _PLUGIN_EXCEPTION_H_

#include "exceptions/Exception.h"

class PluginException {
      private:
         char* val;
      public:
         PluginException(const char *fmt, ... );
   
         char* get_debug_string() {
            return val;
         }
};

#endif // _PLUGIN_EXCEPTION_H_
