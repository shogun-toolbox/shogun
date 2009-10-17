/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __SG_INIT_H__
#define __SG_INIT_H__

#include <stdio.h>

namespace shogun
{
/** This function must be called before libshogun is used. Usually shogun does
 * not provide any output messages (neither debugging nor error; apart from
 * exceptions). This function allows one to specify customized output
 * callback functions and a callback function to check for exceptions:
 *
 * @param print_message function pointer to print a message
 * @param print_warning function pointer to print a warning message
 * @param print_error function pointer to print an error message (this will be
 *                                  printed before shogun throws an exception)
 *
 * @param cancel_computations function pointer to check for exception
 *
 */
void init_shogun(void (*print_message)(FILE* target, const char* str) = NULL,
		void (*print_warning)(FILE* target, const char* str) = NULL,
		void (*print_error)(FILE* target, const char* str) = NULL,
		void (*cancel_computations)(bool &delayed, bool &immediately)=NULL);

/** This function must be called when one stops using libshogun. It will
 * perform a number of cleanups */
void exit_shogun();

/// function called to print normal messages
extern void (*sg_print_message)(FILE* target, const char* str);

/// function called to print warning messages
extern void (*sg_print_warning)(FILE* target, const char* str);

/// function called to print error messages
extern void (*sg_print_error)(FILE* target, const char* str);

/// function called to cancel things
extern void (*sg_cancel_computations)(bool &delayed, bool &immediately);
}
#endif //__SG_INIT__
