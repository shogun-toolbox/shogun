/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Viktor Gal, Yuyu Zhang, 
 *          Thoralf Klein, Evan Shelhamer, Evangelos Anagnostopoulos
 */

#ifndef PARALLEL_H__
#define PARALLEL_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>

namespace shogun
{
/** @brief Class Parallel provides helper functions for multithreading.
 *
 * For example it can be used to determine the number of CPU cores in your
 * computer and is the place where you define the number of CPUs that shall be
 * used in computations.
 */
class SHOGUN_EXPORT Parallel
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

	// FIXME: Should be dropped, but needed to be wrappable by some
	int32_t ref() { return 1; }
	int32_t ref_count() const { return 1; }
	int32_t unref() { return 1; }

private:
	/** number of threads */
	int32_t num_threads;
};
}
#endif
