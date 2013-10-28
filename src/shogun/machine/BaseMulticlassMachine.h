/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef BASEMULTICLASSMACHINE_H__
#define BASEMULTICLASSMACHINE_H__

#include <shogun/machine/Machine.h>

namespace shogun
{

class CMachine;

/** Base class of all Multiclass Machines.
 */
class CBaseMulticlassMachine: public CMachine
{
public:
    /** constructor */
	CBaseMulticlassMachine();

    /** destructor */
	virtual ~CBaseMulticlassMachine();

    /** get name */
    virtual const char* get_name() const { return "BaseMulticlassMachine"; }

	/** get number of machines
	 *
	 * @return number of machines
	 */
	int32_t get_num_machines() const;

	/** get problem type */
	virtual EProblemType get_machine_problem_type() const;

	/** check whether the labels is valid.
	 *
	 * @param lab the labels being checked, guaranteed to be non-NULL
	 */
	virtual bool is_label_valid(CLabels *lab) const;

protected:

	/** machines */
	CDynamicObjectArray *m_machines;
};

} /* shogun */

#endif /* end of include guard: BASEMULTICLASSMACHINE_H__ */

