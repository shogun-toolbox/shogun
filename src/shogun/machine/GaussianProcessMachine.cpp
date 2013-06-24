/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/GaussianProcessMachine.h>

using namespace shogun;

CGaussianProcessMachine::CGaussianProcessMachine()
{
	init();
}

CGaussianProcessMachine::CGaussianProcessMachine(CInferenceMethod* method)
{
	init();
	set_inference_method(method);
}

void CGaussianProcessMachine::init()
{
	m_method=NULL;

	SG_ADD((CSGObject**) &m_method, "inference_method", "Inference Method.",
	    MS_AVAILABLE);
}

CGaussianProcessMachine::~CGaussianProcessMachine()
{
	SG_UNREF(m_method);
}

#endif /* HAVE_EIGEN3 */
