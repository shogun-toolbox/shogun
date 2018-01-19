/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Soumyajit De, Björn Esser
 */

#ifndef JOB_RESULT_H_
#define JOB_RESULT_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>

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
