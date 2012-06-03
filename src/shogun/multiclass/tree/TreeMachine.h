/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef TREEMACHINE_H__
#define TREEMACHINE_H__

#include <shogun/machine/BaseMulticlassMachine.h>
#include <shogun/multiclass/tree/TreeMachineNode.h>

namespace shogun
{

class CTreeMachine: public CBaseMulticlassMachine
{
public:
    /** constructor */
	CTreeMachine();

    /** destructor */
	virtual ~CTreeMachine();

    /** get name */
    virtual const char* get_name() const { return "TreeMachine"; }

private:
	CTreeMachineNode *m_root;

	/** to prevent compile error of class_list.cpp */
	virtual void __placeholder__()=0;
};

} /* shogun */ 

#endif /* end of include guard: TREEMACHINE_H__ */

