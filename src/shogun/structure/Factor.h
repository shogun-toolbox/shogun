/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu 
 * Copyright (C) 2013 Shell Hu 
 */

#ifndef __FACTOR_RELATED_H__
#define __FACTOR_RELATED_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/structure/FactorType.h>

namespace shogun 
{

/** @brief Class CFactorDataSource Source for factor data.
 * In some cases, the same data can be shared by many factors.
 */
class CFactorDataSource : public CSGObject
{
public:
	/** default constructor is prohibitted */
	CFactorDataSource();

	/** constructor 
	 *
	 * @param dense dense factor data
	 */
	CFactorDataSource(SGVector<float64_t> dense);

	/** constructor 
	 *
	 * @param sparse sparse factor data
	 */
	CFactorDataSource(SGSparseVector<float64_t> sparse);

	/** destructor */
	virtual ~CFactorDataSource();

	/** @return class name */
	virtual const char* get_name() const { return "FactorDataSource"; }

	/** @return using sparse data or not */
	virtual bool is_sparse() const;

	/** @return dense data vector */
	virtual SGVector<float64_t> get_data() const;

	/** @return sparse data vector */
	virtual SGSparseVector<float64_t> get_data_sparse() const;

	/** set dense data
	 *
	 * @param dense data vector
	 */
	virtual void set_data(SGVector<float64_t> dense);

	/** set sparse data
	 *
	 * @param sparse pointer to sparse entries
	 * @param dlen number of entries
	 */
	virtual void set_data_sparse(SGSparseVectorEntry<float64_t>* sparse, int32_t dlen);

private:
	/** register parameters */
	void init();

private:
	/** dense data */
	SGVector<float64_t> m_dense;

	/** sparse data */
	SGSparseVector<float64_t> m_sparse;
};

/** @brief Class CFactor A factor is defined on a clique in the factor graph.
 * Each factor can have its own data, either dense, sparse or shared data.
 * Note that currently this class is table factor oriented. 
 */
class CFactor : public CSGObject
{
public:
	/** default constructor */
	CFactor();

	/** Constructor
	 *
	 * @param ftype factor type
	 * @param var_index indices of variables
	 * @param data dense data, can be empty 
	 */
	CFactor(CTableFactorType* ftype, SGVector<int32_t> var_index, SGVector<float64_t> data);

	/** Constructor
	 *
	 * @param ftype factor type
	 * @param var_index indices of variables
	 * @param data_sparse sparse data, can be empty
	 */
	CFactor(CTableFactorType* ftype, SGVector<int32_t> var_index,
		SGSparseVector<float64_t> data_sparse);

	/** Constructor
	 *
	 * @param ftype factor type
	 * @param var_index indices of variables
	 * @param data_source common data for many factors
	 */
	CFactor(CTableFactorType* ftype, SGVector<int32_t> var_index,
		CFactorDataSource* data_source);

	/** deconstructor */
	virtual ~CFactor();

	/** @return class name */
	virtual const char* get_name() const { return "Factor"; }

	/** @return factor type pointer */
	CTableFactorType* get_factor_type() const;

	/** set factor type
	 *
	 * @param ftype factor type
	 */
	void set_factor_type(CTableFactorType* ftype);

	/** @return adjacent variables */
	const SGVector<int32_t> get_variables() const;

	/** set variables
	 * 
	 * @param vars indices of variables
	 */
	void set_variables(SGVector<int32_t> vars);

	/** @return cardinalities of variables */
	const SGVector<int32_t> get_cardinalities() const;

	/** @return dense factor data */
	SGVector<float64_t> get_data() const;

	/** @return sparse factor data, note that call get_dense() to get SGVector */
	SGSparseVector<float64_t> get_data_sparse() const;

	/** set dense data
	 *
	 * @param data_dense data vector
	 */
	void set_data(SGVector<float64_t> data_dense);

	/** set sparse data
	 *
	 * @param data_sparse pointer to sparse entries
	 * @param dlen number of entries
	 */
	void set_data_sparse(SGSparseVectorEntry<float64_t>* data_sparse, int32_t dlen);

	/** @return whether this factor has data */
	bool is_data_dependent() const;

	/** @return whether data vector is sparse */
	bool is_data_sparse() const;

	/** @return energy table which are in Matlab-linearized order: 
	 * leftmost indices run by one.
	 */
	SGVector<float64_t> get_energies() const;

	/** @return energy table which are in Matlab-linearized order: 
	 * leftmost indices run by one.
	 */
	SGVector<float64_t> get_energies();

	/** get an entry in the energy table
	 * @param index in the table
	 * @return energy value
	 */
	float64_t get_energy(int32_t index) const;

	/** set energies with new values 
	 * @param ft_energies new energy table
	 */
	void set_energies(SGVector<float64_t> ft_energies);

	/** set energy for a particular state
	 * @param ei index in the energy table
	 * @param value energy value
	 */
	void set_energy(int32_t ei, float64_t value);

	/** evaluate energy for a given assignment 
	 * @param state variable assignments 
	 * @return energy
	 */
	float64_t evaluate_energy(const SGVector<int32_t> state) const;

	/** Compute energy table */
	void compute_energies();

	/** Compute parameter gradient from marginals and factor data
	 *
	 * @param marginals marginal distribution of the factor
	 * @param parameter_gradient output gradients
	 * @param mult multiplier 
	 */
	void compute_gradients(const SGVector<float64_t> marginals,
		SGVector<float64_t>& parameter_gradient, double mult = 1.0) const;

protected:
	/** factor type */
	CTableFactorType* m_factor_type;

	/** variable indices */
	SGVector<int32_t> m_var_index;

	/** energy table */ 
	SGVector<float64_t> m_energies;

	/** shared data */
	CFactorDataSource* m_data_source;

	/** dense data */
	SGVector<float64_t> m_data;

	/** sparse data */
	SGSparseVector<float64_t> m_data_sparse;

	/** whether the factor is data dependent */
	bool m_is_data_dep;

private:
	/** register & init parameters */
	void init();
};

}

#endif

