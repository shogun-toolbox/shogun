/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _SHOGUN_EXCEPTION_H_
#define _SHOGUN_EXCEPTION_H_

#include <shogun/lib/config.h>

#include <exception>
#include <string>

namespace shogun
{
	/** @brief Class ShogunException defines an exception which is thrown
	 * whenever an
	 * error inside of shogun occurs.
	 */
	class ShogunException : public std::exception
	{
	public:
		/** constructor
		 *
		 * @param str exception string
		 */
		explicit ShogunException(const std::string& what_arg);

		/** constructor
		 *
		 * @param str exception string
		 */
		explicit ShogunException(const char* what_arg);

		/** copy constructor
		 *
		 * @param orig source object
		 */
		ShogunException(const ShogunException& orig);

		/** destructor
		 */
		virtual ~ShogunException();

		virtual const char* what() const noexcept override;

	protected:
		/** exception string */
		std::string msg;
	};
}
#endif // _SHOGUN_EXCEPTION_H_
