%{
 #include <shogun/base/init.h>

 extern void sg_global_print_message(FILE* target, const char* str);
 extern void sg_global_print_warning(FILE* target, const char* str);
 extern void sg_global_print_error(FILE* target, const char* str);
#ifndef DISABLE_CANCEL_CALLBACK
 extern void sg_global_cancel_computations(bool &delayed, bool &immediately);
#endif
%}

%init %{
#ifndef DISABLE_CANCEL_CALLBACK
    init_shogun(&sg_global_print_message, &sg_global_print_warning,
            &sg_global_print_error, &sg_global_cancel_computations);
#else
    init_shogun(&sg_global_print_message, &sg_global_print_warning,
            &sg_global_print_error);
#endif
%}
