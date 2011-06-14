/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "modelselection/ModelSelectionParametersRootNode.h"

using namespace shogun;

CModelSelectionParametersRootNode::CModelSelectionParametersRootNode()
{

}

CModelSelectionParametersRootNode::~CModelSelectionParametersRootNode()
{

}

void CModelSelectionParametersRootNode::print(const char* prefix)
{
	SG_PRINT("%sroot node\n", prefix);
	CModelSelectionParametersTree::print(prefix);
}
