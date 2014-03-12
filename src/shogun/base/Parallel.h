/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef PARALLEL_H__
#define PARALLEL_H__

#include <shogun/lib/common.h>

namespace shogun
{
class RefCount;
/** @brief Class Parallel provides helper functions for multithreading.
 *
 * For example it can be used to determine the number of CPU cores in your
 * computer and is the place where you define the number of CPUs that shall be
 * used in computations.
 */
class Parallel
{
public:
	/** constructor */
	Parallel();

	/** copy constructor */
	Parallel(const Parallel& orig);

	/** destructor */
	virtual ~Parallel();

	/** get num of cpus
	 * @return number of CPUs
	 */
	int32_t get_num_cpus() const;

	/** set number of threads
	 * @param n number of threads
	 */
	void set_num_threads(int32_t n);

	/** get number of threads
	 * @return number of threads
	 */
	int32_t get_num_threads() const;

	/** ref
	 * @return current ref counter
	 */
	int32_t ref();

	/** get ref count
	 * @return current ref counter
	 */
	int32_t ref_count() const;

	/** unref
	 * @return current ref counter
	 */
	int32_t unref();

private:
	/** ref counter */
	RefCount* m_refcount;

	/** number of threads */
	int32_t num_threads;
};
}
#endif
