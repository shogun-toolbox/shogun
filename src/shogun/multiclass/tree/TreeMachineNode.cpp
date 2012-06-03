/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/tree/TreeMachineNode.h>

using namespace shogun;

CTreeMachineNode::CTreeMachineNode()
	:m_left(NULL), m_right(NULL), m_machine(-1)
{
	SG_ADD((CSGObject**)&m_left,"m_left", "Left subtree", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_right,"m_right", "Right subtree", MS_NOT_AVAILABLE);
	SG_ADD(&m_machine,"m_machine", "Index of associated machine", MS_NOT_AVAILABLE);
}

CTreeMachineNode::~CTreeMachineNode()
{
	SG_UNREF(m_left);
	SG_UNREF(m_right);
}
