/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu 
 * Copyright (C) 2013 Shell Hu 
 */

#ifndef __FACTORGRAPH_H__
#define __FACTORGRAPH_H__

#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/Features.h>
#include <shogun/structure/Factor.h>
#include <shogun/structure/FactorGraphLabels.h>

namespace shogun {

/* @brief Class CFactorGraph a factor graph is a structured input in general 
 */
class CFactorGraph : public CFeatures
{
public:
	CFactorGraph();

	/** Constructor 
	 *
	 * @param card cardinalities of all the variables in the factor graph 
	 */
	CFactorGraph(const SGVector<int32_t> card);

	/** Copy constructor 
	 *
	 * @param fg a factor graph instance
	 */
	CFactorGraph(const CFactorGraph &fg);

	/** deconstructor */
	~CFactorGraph();

	/** @return class name */
	virtual const char* get_name() const { return "FactorGraph"; }

	/** @return all the factors */
	CDynamicObjectArray* get_factors() const;

	/** @return cardinalities */
	SGVector<int32_t> get_cardinalities() const;

	/** set cardinalities 
	 *
	 * @param cards cardinalities of all variables
	 */
	void set_cardinalities(SGVector<int32_t> cards);

	/** @return all the shared data */
	CDynamicObjectArray* get_factor_data_sources() const;

	/** add a factor 
	 *
	 * @param factor a factor pointer
	 */
	void add_factor(CFactor* factor);

	/** add a data source  
	 *
	 * @param datasource a factor data source 
	 */
	void add_data_source(CFactorDataSource* datasource);

	/** compute energy tables in the factor graph */ 
	void compute_energies();

	/** evaluate energy given full assignment 
	 *
	 * @param state an assignment 
	 */
	float64_t evaluate_energy(const SGVector<int32_t> state) const;

	/** evaluate energy for a given fully observed assignment
	 *
	 * @param obs factor graph observation 
	 */
	float64_t evaluate_energy(const CFactorGraphObservation* obs) const;

	/** @return copy of factor graph */
	virtual CFactorGraph* duplicate() const;

	/** @return feature tyep */
	virtual EFeatureType get_feature_type() const {return F_ANY;}

	/** @return feature class */
	virtual EFeatureClass get_feature_class() const {return C_ANY;}

	/** @return number of factors */
	virtual int32_t get_num_vectors() const;

private:

	void register_parameters();

private:

	SGVector<int32_t> m_cardinalities;

	CDynamicObjectArray* m_factors;

	CDynamicObjectArray* m_datasources;
};

}

#endif

