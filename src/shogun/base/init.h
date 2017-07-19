/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Pan Deng, Heiko Strathmann, Soeren Sonnenburg, Giovanni De Toni, 
 *          Yuyu Zhang, Viktor Gal, Sergey Lisitsyn
 */

#ifndef __SG_INIT_H__
#define __SG_INIT_H__

#include <shogun/lib/common.h>
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

	/** @return the globally over-ridden floating point epsilon for
	 * CMath::fequals
	 */
	float64_t get_global_fequals_epsilon();

	/** Globally over-ride the floating point epsilon for CMath::fequals.
	 * Hack required for CSGObject::equals checks for certain serialization
	 * formats.
	 * @param fequals_epsilon new epsilon to use
	 */
	void set_global_fequals_epsilon(float64_t fequals_epsilon);

	/** @return whether global linient check for CMath::fequals is enabled
	 */
	bool get_global_fequals_tolerant();

	/** Globally enable linient check for CMath::fequals.
	 * Hack required for CSGObject::equals checks for certain serialization
	 * formats.
	 * @param fequals_tolerant whether or not to use tolerant check
	 */
	void set_global_fequals_tolerant(bool fequals_tolerant);

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

/** Set global random seed
 * @param seed seed for random generator
 */
void set_global_seed(uint32_t seed);

/** get global random seed
 * @return random seed
 */
uint32_t get_global_seed();

uint32_t generate_seed();

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
