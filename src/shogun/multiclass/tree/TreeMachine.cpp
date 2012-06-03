/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/tree/TreeMachine.h>

using namespace shogun;

CTreeMachine::CTreeMachine()
	:m_root(NULL) 
{
	SG_ADD((CSGObject**)&m_root,"m_root", "tree structure", MS_NOT_AVAILABLE);
}

CTreeMachine::~CTreeMachine()
{
	SG_UNREF(m_root);
}
