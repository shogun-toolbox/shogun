#ifndef _SHOGUN_EXCEPTION_H_
#define _SHOGUN_EXCEPTION_H_

class ShogunException {
    public:
        ShogunException(const char* str);

        inline const char* get_exception_string() {
            return val;
        }
    private:
        char* val;
};

#endif // _SHOGUN_EXCEPTION_H_
