/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Christian Gehl
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST
 */

#include "classifier/DistanceMachine.h"

using namespace shogun;

CDistanceMachine::CDistanceMachine()
: CClassifier(), distance(NULL)
{
}

CDistanceMachine::~CDistanceMachine()
{
	SG_UNREF(distance);
}
