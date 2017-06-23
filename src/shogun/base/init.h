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

#include <shogun/lib/config.h>

#include <functional>
#include <stdio.h>

namespace shogun
{
	class SGIO;
	class CMath;
	class Version;
	class Parallel;
	class CRandom;
	class SGLinalg;
	class CSignal;

	/** This function must be called before libshogun is used. Usually shogun
	 * does
	 * not provide any output messages (neither debugging nor error; apart from
	 * exceptions). This function allows one to specify customized output
	 * callback functions and a callback function to check for exceptions:
	 *
	 * @param print_message function pointer to print a message
	 * @param print_warning function pointer to print a warning message
	 * @param print_error function pointer to print an error message (this will
	 * be
	 *                                  printed before shogun throws an
	 * exception)
	 *
	 * @param cancel_computations function pointer to check for exception
	 *
	 */
	void init_shogun(
	    const std::function<void(FILE*, const char*)> print_message = nullptr,
	    const std::function<void(FILE*, const char*)> print_warning = nullptr,
	    const std::function<void(FILE*, const char*)> print_error = nullptr);

	/** init shogun with defaults */
	void init_shogun_with_defaults();

	/** This function must be called when one stops using libshogun. It will
	 * perform a number of cleanups */
	void exit_shogun();

	/** set the global io object
	 *
	 * @param io io object to use
	 */
	void set_global_io(SGIO* io);

	/** get the global io object
	 *
	 * @return io object
	 */
	SGIO* get_global_io();

	/** set the global parallel object
	 *
	 * @param parallel parallel object to use
	 */
	void set_global_parallel(Parallel* parallel);

	/** get the global parallel object
	 *
	 * @return parallel object
	 */
	Parallel* get_global_parallel();

	/** set the global version object
	 *
	 * @param version version object to use
	 */
	void set_global_version(Version* version);

	/** get the global version object
	 *
	 * @return version object
	 */
	Version* get_global_version();

	/** set the global math object
	 *
	 * @param math math object to use
	 */
	void set_global_math(CMath* math);

	/** get the global math object
	 *
	 * @return math object
	 */
	CMath* get_global_math();

	/** set the global random object
	 *
	 * @param rand random object to use
	 */
	void set_global_rand(CRandom* rand);

	/** get the global random object
	 *
	 * @return random object
	 */
	CRandom* get_global_rand();

#ifndef SWIG // SWIG should skip this part
/** get the global linalg library object
 *
 * @return linalg object
 */
SGLinalg* get_global_linalg();
#endif

/** get the global singnal handler object
 *
 * @return linalg object
 */
CSignal* get_global_signal();

/** Checks environment variables and modifies global objects
 */
void init_from_env();

/// function called to print normal messages
extern std::function<void(FILE*, const char*)> sg_print_message;

/// function called to print warning messages
extern std::function<void(FILE*, const char*)> sg_print_warning;

/// function called to print error messages
extern std::function<void(FILE*, const char*)> sg_print_error;
}
#endif //__SG_INIT__
