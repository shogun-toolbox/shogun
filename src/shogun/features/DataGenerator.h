/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __DATAGENERATOR_H_
#define __DATAGENERATOR_H_

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/** @brief Class that is able to generate various data samples, which may be
 * used for examples in SHOGUN.
 */
class CDataGenerator: public CSGObject
{
public:
	CDataGenerator();

	virtual ~CDataGenerator();

	/** produces a set of vectors, where each dimension is standard normally
	 * distributed and the mean of the first dimension is shifted by a
	 * provided value
	 *
	 * @param m number of samples to generate
	 * @param dim dimension of generated samples
	 * @param mean_shift is added mean of first dimension
	 * @return matrix with samples
	 */
	SGMatrix<float64_t> generate_mean_data(index_t m, index_t dim,
			float64_t mean_shift);

	inline virtual const char* get_name() const { return "DataGenerator"; }

private:
	/** registers all parameters and initializes variables with defaults */
	void init();

};

}

#endif /* __DATAGENERATOR_H_ */
