/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Parijat Mazumdar
 */

#ifndef _LKMEANS_H__
#define _LKMEANS_H__

#include <stdio.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{
	void lKMeans(int32_t k, CDistance* distance, int32_t max_iter, SGMatrix<float64_t> mus, bool fixed_centers);
}
#endif
