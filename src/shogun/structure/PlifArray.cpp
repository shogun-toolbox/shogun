/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Bjoern Esser
 */

#include <shogun/lib/config.h>

#include <stdio.h>
#include <string.h>

#include <shogun/io/SGIO.h>

#include <shogun/structure/PlifArray.h>
#include <shogun/structure/Plif.h>

//#define PLIFARRAY_DEBUG

using namespace shogun;

PlifArray::PlifArray()
: PlifBase()
{
	min_value=-1e6;
	max_value=1e6;
}

PlifArray::~PlifArray()
{
}

void PlifArray::add_plif(const std::shared_ptr<PlifBase>& new_plif)
{
	ASSERT(new_plif)
	m_array.push_back(new_plif) ;

	min_value = -1e6 ;
	for (int32_t i=0; i<m_array.size(); i++)
	{
		ASSERT(m_array[i])
		if (!m_array[i]->uses_svm_values())
			min_value = Math::max(min_value, m_array[i]->get_min_value()) ;
	}

	max_value = 1e6 ;
	for (int32_t i=0; i<m_array.size(); i++)
		if (!m_array[i]->uses_svm_values())
			max_value = Math::min(max_value, m_array[i]->get_max_value()) ;
}

void PlifArray::clear()
{
	m_array.clear();
	min_value = -1e6 ;
	max_value = 1e6 ;
}

float64_t PlifArray::lookup_penalty(
	float64_t p_value, float64_t* svm_values) const
{
	//min_value = -1e6 ;
	//max_value = 1e6 ;
	if (p_value<min_value || p_value>max_value)
	{
		//io::warn("lookup_penalty: p_value: {} min_value: {}, max_value: {}",p_value, min_value, max_value);
		return -Math::INFTY ;
	}
	float64_t ret = 0.0 ;
	for (int32_t i=0; i<m_array.size(); i++)
		ret += m_array[i]->lookup_penalty(p_value, svm_values) ;
	return ret ;
}

float64_t PlifArray::lookup_penalty(
	int32_t p_value, float64_t* svm_values) const
{
	//min_value = -1e6 ;
	//max_value = 1e6 ;
	if (p_value<min_value || p_value>max_value)
	{
		//io::warn("lookup_penalty: p_value: {} min_value: {}, max_value: {}",p_value, min_value, max_value);
		return -Math::INFTY ;
	}
	float64_t ret = 0.0 ;
	for (int32_t i=0; i<m_array.size(); i++)
	{
		float64_t val = m_array[i]->lookup_penalty(p_value, svm_values) ;
		ret += val ;
#ifdef PLIFARRAY_DEBUG
		CPlif * plif = (CPlif*)m_array[i] ;
		if (plif->get_use_svm())
			io::print("penalty[{}]={:1.5f} (use_svm={} -> {:1.5f})\n", i, val, plif->get_use_svm(), svm_values[plif->get_use_svm()-1]);
		else
			io::print("penalty[{}]={:1.5f}\n", i, val);
#endif
	}
	return ret ;
}

void PlifArray::penalty_clear_derivative()
{
	for (int32_t i=0; i<m_array.size(); i++)
		m_array[i]->penalty_clear_derivative() ;
}

void PlifArray::penalty_add_derivative(
	float64_t p_value, float64_t* svm_values, float64_t factor)
{
	for (int32_t i=0; i<m_array.size(); i++)
		m_array[i]->penalty_add_derivative(p_value, svm_values, factor) ;
}

bool PlifArray::uses_svm_values() const
{
	for (int32_t i=0; i<m_array.size(); i++)
		if (m_array[i]->uses_svm_values())
			return true ;
	return false ;
}

int32_t PlifArray::get_max_id() const
{
	int32_t max_id = 0 ;
	for (int32_t i=0; i<m_array.size(); i++)
		max_id = Math::max(max_id, m_array[i]->get_max_id()) ;
	return max_id ;
}

void PlifArray::get_used_svms(int32_t* num_svms, int32_t* svm_ids)
{
	io::print("get_used_svms: num: {} \n",m_array.size());
	for (int32_t i=0; i<m_array.size(); i++)
	{
		m_array[i]->get_used_svms(num_svms, svm_ids);
	}
	io::print("\n");
}
