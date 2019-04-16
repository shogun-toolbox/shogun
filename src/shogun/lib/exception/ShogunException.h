/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _SHOGUN_EXCEPTION_H_
#define _SHOGUN_EXCEPTION_H_

#include <shogun/lib/config.h>

#include <stdexcept>
#include <string>

namespace shogun
{
	/** @brief Class ShogunException defines an exception which is thrown
	 * whenever an error inside of shogun occurs.
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

		const char* what() const noexcept override;

	private:
		/** exception object */
		std::runtime_error msg;
	};
} // namespace shogun
#endif // _SHOGUN_EXCEPTION_H_
