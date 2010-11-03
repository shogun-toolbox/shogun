/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SHOGUN_EXCEPTION_H_
#define _SHOGUN_EXCEPTION_H_

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
