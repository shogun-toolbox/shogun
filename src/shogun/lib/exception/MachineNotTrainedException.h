/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#ifndef _MACHINE_NOT_TRAINED_EXCEPTION_H_
#define _MACHINE_NOT_TRAINED_EXCEPTION_H_

#include <shogun/lib/exception/ShogunException.h>

namespace shogun
{

	/** @brief Class MachineNotTrainedException defines an exception which is
	 * thrown whenever a machine or a transformer in Shogun used but haven't be
	 * fitted.
	 */
	class MachineNotTrainedException : public ShogunException
	{
	public:
		/** constructor
		*
		* @param str exception string
		*/
		explicit MachineNotTrainedException(const std::string& what_arg)
		    : ShogunException(what_arg)
		{
		}

		/** constructor
		 *
		 * @param str exception string
		 */
		explicit MachineNotTrainedException(const char* what_arg)
		    : ShogunException(what_arg)
		{
		}
	};
}

#endif
