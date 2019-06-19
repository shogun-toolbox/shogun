/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#ifndef _SHOGUN_NOT_IMPLEMENTED_EXCEPTION_H_
#define _SHOGUN_NOT_IMPLEMENTED_EXCEPTION_H_

#include <shogun/lib/exception/ShogunException.h>

namespace shogun
{
/** @brief Class ShogunNotImplementedException defines an exception which is thrown whenever an
 * error inside of shogun occurs.
 */
class ShogunNotImplementedException: public ShogunException
{
	public:
		/** constructor
		 *
		 * @param str exception string
		 */
		explicit ShogunNotImplementedException(const std::string& what_arg)
			: ShogunException(what_arg) {}

		/** constructor
		 *
		 * @param str exception string
		 */
		explicit ShogunNotImplementedException(const char* what_arg)
			: ShogunException(what_arg) {}


		/** destructor
		 */
        virtual ~ShogunNotImplementedException();


	private:
		/** exception string */
		std::string msg;
};
}
#endif // _SHOGUN_NOT_IMPLEMENTED_EXCEPTION_H_
