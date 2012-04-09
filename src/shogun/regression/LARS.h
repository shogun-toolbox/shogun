/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef LARS_H_WSBLKNVU
#define LARS_H_WSBLKNVU

#include <shogun/machine/LinearMachine.h>

namespace shogun 
{

class LARS: public CLinearMachine
{
public:
	LARS();
	virtual ~LARS();

protected:
	virtual bool train_machine(CFeatures* data=NULL);
}; // class LARS

} // namespace shogun

#endif /* end of include guard: LARS_H_WSBLKNVU */

