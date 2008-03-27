/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Christian Gehl
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST
 */

#include "distance/DistanceMachine.h"

CDistanceMachine::CDistanceMachine():CClassifier(),distance(NULL)
{
}

CDistanceMachine::~CDistanceMachine()
{
	SG_UNREF(distance);
}
