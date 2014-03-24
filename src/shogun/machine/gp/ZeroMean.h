/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CZEROMEAN_H_
#define CZEROMEAN_H_

#include <shogun/lib/config.h>
#include <shogun/machine/gp/MeanFunction.h>

namespace shogun
{

/** @brief The zero mean function class.
 *
 * Simple mean function that assumes a mean of zero.
 */
class CZeroMean : public CMeanFunction
{
public:
	/** constructor */
	CZeroMean();

	virtual ~CZeroMean();

	/** returns name of the mean function
	 *
	 * @return name ZeroMean
	 */
	virtual const char* get_name() const { return "ZeroMean"; }

	/** returns the mean of the specified data
	 *
	 * @param features data points arranged in a matrix with rows representing the number
	 * of features
	 *
	 * @return mean of feature vectors
	 */
	virtual SGVector<float64_t> get_mean_vector(const CFeatures* features) const;
};
}
#endif /* CZEROMEAN_H_ */
