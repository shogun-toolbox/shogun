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

#include <shogun/machine/MulticlassMachine.h>
#include <shogun/machine/tree/TreeMachineNode.h>

namespace shogun
{

class CTreeMachine: public CMulticlassMachine
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
};

} /* shogun */ 

#endif /* end of include guard: TREEMACHINE_H__ */

