/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#ifndef _INVALID_STATE_EXCEPTION_H_
#define _INVALID_STATE_EXCEPTION_H_

#include <shogun/lib/exception/ShogunException.h>

namespace shogun
{

	/** @brief Class InvalidStateException defines an exception which is thrown
	 * whenever an object in Shogun in invalid state is used.
	 */
	class InvalidStateException : public ShogunException
	{
	public:
		/** constructor
		*
		* @param str exception string
		*/
		explicit InvalidStateException(const std::string& what_arg)
		    : ShogunException(what_arg)
		{
		}

		/** constructor
		 *
		 * @param str exception string
		 */
		explicit InvalidStateException(const char* what_arg)
		    : ShogunException(what_arg)
		{
		}
	};
}

#endif
