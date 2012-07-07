/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CZEROMEAN_H_
#define CZEROMEAN_H_

#include <shogun/regression/gp/MeanFunction.h>

namespace shogun
{

/** @brief Zero Mean Function
 *
 *	Simple mean function that assumes a mean
 *	of zero.
 *
 */
class CZeroMean: public CMeanFunction
{

public:

	/*Constructor*/
	CZeroMean();

	/*Destructor*/
	virtual ~CZeroMean();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	inline virtual const char* get_name() const { return "ZeroMean"; }

	/** Returns the mean of the specified data
	 *
	 * @param data points arranged in a matrix with rows representing
	 * the number of features
	 *
	 * @return Mean of feature vectors
	 */
	virtual SGVector<float64_t> get_mean_vector(SGMatrix<float64_t>& data);

};

}

#endif /* CZEROMEAN_H_ */
