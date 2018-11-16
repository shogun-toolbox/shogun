/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg
 */

#include <shogun/lib/common.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>

using namespace shogun;

CMeanShiftDataGenerator::CMeanShiftDataGenerator() :
		CStreamingDenseFeatures<float64_t>()
{
	init();
}

CMeanShiftDataGenerator::CMeanShiftDataGenerator(float64_t mean_shift,
		index_t dimension, index_t dimension_shift) :
				CStreamingDenseFeatures<float64_t>()
{
	init();
	set_mean_shift_model(mean_shift, dimension, dimension_shift);
}

CMeanShiftDataGenerator::~CMeanShiftDataGenerator()
{
}

void CMeanShiftDataGenerator::set_mean_shift_model(float64_t mean_shift,
		index_t dimension, index_t dimension_shift)
{
	REQUIRE(dimension_shift<dimension, "%s::set_mean_shift_model(%f,%d,%d): "
			"Dimension of shift is larger than number of dimensions!\n",
			mean_shift, dimension, dimension_shift);

	m_dimension=dimension;
	m_mean_shift=mean_shift;
	m_dimension_shift=dimension_shift;
}

void CMeanShiftDataGenerator::init()
{
	SG_ADD(&m_dimension, "dimension", "Dimension of data");
	SG_ADD(&m_mean_shift, "mean_shift", "Mean shift in one dimension");
	SG_ADD(&m_dimension_shift, "m_dimension_shift", "Dimension of mean shift");

	m_dimension=0;
	m_mean_shift=0;
	m_dimension_shift=0;

	unset_generic();
}

bool CMeanShiftDataGenerator::get_next_example()
{
	SG_SDEBUG("entering\n");

	/* allocate space */
	SGVector<float64_t> result=SGVector<float64_t>(m_dimension);

	/* fill with std normal data */
	for (index_t i=0; i<m_dimension; ++i)
		result[i]=CMath::randn_double();

	/* mean shift in selected dimension */
	result[m_dimension_shift]+=m_mean_shift;

	/* save example back to superclass */
	CMeanShiftDataGenerator::current_vector=result;

	SG_SDEBUG("leaving\n");
	return true;
}

void CMeanShiftDataGenerator::release_example()
{
	SGVector<float64_t> temp=SGVector<float64_t>();
	CMeanShiftDataGenerator::current_vector=temp;
}
