/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _SHOGUN_EXCEPTION_H_
#define _SHOGUN_EXCEPTION_H_

#include <shogun/lib/config.h>

namespace shogun
{
/** @brief Class ShogunException defines an exception which is thrown whenever an
 * error inside of shogun occurs.
 */
class ShogunException
{
	void init(const char* str);

	public:
		/** constructor
		 *
		 * @param str exception string
		 */
		ShogunException(const char* str);

		/** copy constructor
		 *
		 * @param orig source object
		 */
		ShogunException(const ShogunException& orig);

		/** destructor
		 */
        virtual ~ShogunException();

		/** get exception string
		 *
		 * @return the exception string
		 */
		inline const char* get_exception_string() { return val; }

	private:
		/** exception string */
		char* val;
};
}
#endif // _SHOGUN_EXCEPTION_H_
