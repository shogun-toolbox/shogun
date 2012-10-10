/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>

using namespace shogun;

template<class T>
CMeanShiftDataGenerator<T>::CMeanShiftDataGenerator() :
		CStreamingDenseFeatures<T>()
{
	init();
}

template<class T>
CMeanShiftDataGenerator<T>::CMeanShiftDataGenerator(T mean_shift,
		index_t dimension): CStreamingDenseFeatures<T>()
{
	init();
	set_mean_shift_model(mean_shift, dimension);
}

template<class T>
CMeanShiftDataGenerator<T>::~CMeanShiftDataGenerator()
{
}

template<class T>
void CMeanShiftDataGenerator<T>::set_mean_shift_model(T mean_shift,
		index_t dimension)
{
	m_dimension=dimension;
	m_mean_shift=mean_shift;
}

template<class T>
void CMeanShiftDataGenerator<T>::init()
{
	m_dimension=0;
	m_mean_shift=0;
}

template<class T>
bool CMeanShiftDataGenerator<T>::get_next_example()
{
	SG_SDEBUG("entering CMeanShiftDataGenerator::get_next_example()\n");

	/* allocate space */
	SGVector<T> result=SGVector<T>(m_dimension);

	/* fill with std normal data */
	for (index_t i=0; i<m_dimension; ++i)
		result[i]=CMath::randn_double();

	/* mean shift in first dimension */
	result[0]+=m_mean_shift;

	/* save example back to superclass */
	CMeanShiftDataGenerator<T>::current_vector=result;

	SG_SDEBUG("leaving CMeanShiftDataGenerator::get_next_example()\n");
	return true;
}

template<class T>
void CMeanShiftDataGenerator<T>::release_example()
{
	SGVector<T> temp=SGVector<T>();
	CMeanShiftDataGenerator<T>::current_vector=temp;
}

template class CMeanShiftDataGenerator<bool>;
template class CMeanShiftDataGenerator<char>;
template class CMeanShiftDataGenerator<int8_t>;
template class CMeanShiftDataGenerator<uint8_t>;
template class CMeanShiftDataGenerator<int16_t>;
template class CMeanShiftDataGenerator<uint16_t>;
template class CMeanShiftDataGenerator<int32_t>;
template class CMeanShiftDataGenerator<uint32_t>;
template class CMeanShiftDataGenerator<int64_t>;
template class CMeanShiftDataGenerator<uint64_t>;
template class CMeanShiftDataGenerator<float32_t>;
template class CMeanShiftDataGenerator<float64_t>;
template class CMeanShiftDataGenerator<floatmax_t>;
