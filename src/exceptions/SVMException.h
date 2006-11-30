#ifndef _SVM_EXCEPTION_H_
#define _SVM_EXCEPTION_H_

class SVMException {
      private:
         char *mes;
      public:
         SVMException(const char *_mes) {
            mes = new char[strlen(_mes)];
            strcpy(mes,_mes);
         }

         char* get_debug_string() {
            return mes;
         }
};

#endif // _SVM_EXCEPTION_H_
