/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef JOB_RESULT_H_
#define JOB_RESULT_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{

/** @brief Base class that stores the result of an independent job */
class CJobResult : public CSGObject
{
public:
	/** default constructor */
	CJobResult()
	: CSGObject()
	{
		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** destructor */
	virtual ~CJobResult()
	{
		SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "JobResult";
	}
};

}

#endif // JOB_RESULT_H_
