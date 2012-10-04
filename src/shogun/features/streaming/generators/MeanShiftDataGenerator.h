/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __STREAMINGDENSEDATAGENERATOR_H_
#define __STREAMINGDENSEDATAGENERATOR_H_

#include <shogun/features/streaming/StreamingDenseFeatures.h>

namespace shogun
{


/** @brief
 * Class to generate dense features data via the streaming features interface.
 * The core are pairs of methods to
 * a) set the data model and parameters, and
 * b) to generate a data vector using these model parameters
 * Both methods are automatically called when calling get_next_example()
 * This allows to treat generated data as a stream via the standard streaming
 * features interface
 */
template <class T> class CMeanShiftDataGenerator:
	public CStreamingDenseFeatures<T>
{
public:
	/** Constructor */
	CMeanShiftDataGenerator();

	CMeanShiftDataGenerator(T mean_shift, index_t dim);

	/** Destructor */
	virtual ~CMeanShiftDataGenerator();

	/** @return name of SG_SERIALIZABLE */
	inline virtual const char* get_name() const
	{
		return "MeanShiftDataGenerator";
	}

	void set_mean_shift_model(T mean_shift, index_t dimension);

	bool get_next_example();

private:
	/** registers all parameters and initializes variables with defaults */
	void init();

protected:
	/** model of data to generate */
	T m_mean_shift;
	index_t m_dimension;
};

}

#endif /* __STREAMINGDENSEDATAGENERATOR_H_ */
