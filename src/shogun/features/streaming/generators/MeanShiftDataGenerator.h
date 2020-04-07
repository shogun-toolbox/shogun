/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef __MEANSHIFTDATAGENERATOR_H_
#define __MEANSHIFTDATAGENERATOR_H_

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{


/** Class to generate dense features data via the streaming features interface.
 * The core are pairs of methods to
 * a) set the data model and parameters, and
 * b) to generate a data vector using these model parameters
 * Both methods are automatically called when calling get_next_example()
 * This allows to treat generated data as a stream via the standard streaming
 * features interface.
 *
 * Streaming based data generator that samples from a distribution that is a
 * multivariate isotropic Gaussian in a certain dimensions, where one selected
 * dimension has its mean shifted by some value.
 */
class MeanShiftDataGenerator: public RandomMixin<StreamingDenseFeatures<float64_t>>
{
public:
	/** Constructor */
	MeanShiftDataGenerator();

	/** Constructor
	 *
	 * @param mean_shift Selected dimension is shifted by this amount
	 * @param dimension Number of dimensions
	 * @param m_dimension_shift Dimension that gets shifted
	 */
	MeanShiftDataGenerator(float64_t mean_shift, index_t dimension,
			index_t m_dimension_shift=0);

	/** Destructor */
	~MeanShiftDataGenerator() override;

	/** @return name of SG_SERIALIZABLE */
	const char* get_name() const override
	{
		return "MeanShiftDataGenerator";
	}

	/** Set the mean shift model
	 *
	 * @param mean_shift Selected dimension is shifted by this amount
	 * @param dimension Number of dimensions
	 * @param m_dimension_shift Dimension that gets shifted
	 */
	void set_mean_shift_model(float64_t mean_shift, index_t dimension,
			index_t m_dimension_shift=0);

	/** get the next example from stream */
	bool get_next_example() override;

	/** release the example when done w/ processing */
	void release_example() override;

private:
	/** registers all parameters and initializes variables with defaults */
	void init();

protected:
	/** model of data to generate */
	float64_t m_mean_shift;

	/** dimension */
	index_t m_dimension;

	/** Dimension that is shifted */
	index_t m_dimension_shift;
};

}

#endif /* __MEANSHIFTDATAGENERATOR_H_ */
