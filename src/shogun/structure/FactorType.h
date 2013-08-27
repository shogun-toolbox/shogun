/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu 
 * Copyright (C) 2013 Shell Hu 
 */

#ifndef __FACTOR_TYPE_H__
#define __FACTOR_TYPE_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGString.h>

namespace shogun
{

/* @brief Class CFactorType defines the way of factor parameterization
 */
class CFactorType : public CSGObject
{
public:
	/** default constructor is prohibitted */
	CFactorType();

	/** Constructor
	 * if w.size() == 0
	 *   data_size = prod_card, i.e. product of cardinalities
	 * else if w.size() != 0, i.e. parameterized factor type
	 *   data_size = w.size() / prod_card, ensure divisible
	 *   Note w.size() == prod_card may be data indep case
	 *
	 * @param id used for query in FactorGraphModel
	 * @param card cardinalities of variables in the clique
	 * @param w factor parameters
	 */
	CFactorType(const SGString<char> id, const SGVector<int32_t> card,
		const SGVector<float64_t> w = SGVector<float64_t>());

	/** deconstructor */
	virtual ~CFactorType();

	/** @return name of class */
	virtual const char* get_name() const { return "FactorType"; }

	/** @return get factor type id */
	virtual const char* get_type_id() const;

	/** set factor type id
	 *
	 * @param tid_str a string 
	 * @param length of tid_str
	 */
	virtual void set_type_id(char* tid_str, int32_t slen);

	/** @return get factor parameters */
	virtual SGVector<float64_t> get_w();

	/** @return get factor parameters */
	virtual const SGVector<float64_t> get_w() const;

	/** set parameters
	 *
	 * @param w factor parameters
	 */
	void set_w(SGVector<float64_t> w);

	/** @return dimension of the factor parameter vector */
	virtual int32_t get_w_dim() const;

	/** @return cardinalities of the variables */
	virtual const SGVector<int32_t> get_cardinalities() const;

	/** set cardinalities 
	 *
	 * @param cards cardinalities of variables
	 */
	virtual void set_cardinalities(SGVector<int32_t> cards);

	/** @return product of cardinalities */
	virtual int32_t get_prodcardinalities() const;

	/** parameters mapping to the global model parameters
	 *
	 * @param map indices in model parameter vector
	 */
	virtual void set_params_mapping(SGVector<int32_t> map);

	/** @return is it a table factor type */
	virtual bool is_table() const { return false; }

protected:
	/** initialize m_prod_card and m_cumprod_cards */
	void init_card();

private:
	void init();

protected:
	SGString<char> m_type_id;

	/** variable information */
	SGVector<int32_t> m_cards;
	SGVector<int32_t> m_cumprod_cards;
	int32_t m_prod_card;

	/** data information */
	int32_t m_data_size;

	/** factor paramters */
	SGVector<float64_t> m_w;
	/** indices to model parameters of FactorGraphModel */
	SGVector<int32_t> m_map_params;
};

/* @brief Class CTableFactorType the way that store assignments of variables
 * and energies in a table or a multi-array
 */
class CTableFactorType : public CFactorType
{
public:
	/** default constructor */
	CTableFactorType();

	/** constructor
	 *
	 * @param id used for query in FactorGraphModel
	 * @param card cardinalities of variables in the clique
	 * @param w factor parameters
	 */
	CTableFactorType(const SGString<char> id, const SGVector<int32_t> card,
		const SGVector<float64_t> w = SGVector<float64_t>());

	/** deconstructor */
	virtual ~CTableFactorType();

	/** @return class name */
	virtual const char* get_name() const { return "TableFactorType"; }

	/** @return this is the table implementation but memory inefficient */
	virtual bool is_table() const { return true; }

	/** infer variable state from a given index of energy table
	 * 
	 * @param ei the energy index. 
	 * @param var_index the variable index of this factor:
	 * @return variable state 
	 */
	int32_t state_from_index(int32_t ei, int32_t var_index) const;

	/** get variable assignment from absolute energy index
	 *
	 * @param ei energy index
	 * @return variable assignment
	 */
	SGVector<int32_t> assignment_from_index(int32_t ei) const;

	/** update energy index by substituting with contribution 
	 * from new state of a particular variable
	 * 
	 * @param old_ei old energy index. 
	 * @param var_index the variable index of this factor:
	 * @param var_state new variable state 
	 * @return new energy index 
	 */
	int32_t index_from_new_state(int32_t old_ei, int32_t var_index, 
		int32_t var_state) const;

	/** energy index from a given assignment
	 *
	 * @param assig variable assignments
	 * @return energy index
	 */
	int32_t index_from_assignment(const SGVector<int32_t> assig) const;

	/** energy index in the table of a factor given an universe assignment
	 *
	 * @param assig variable assignments
	 * @param variable indices of that particular factor
	 * @return energy index
	 */
	int32_t index_from_universe_assignment(const SGVector<int32_t> assig,
		const SGVector<int32_t> var_index) const;

	/** compute energy values from parameters for a specific factor.
	 *
	 * @param factor_data dense factor data vector 
	 * @param energies forwarded energy table
	 */
	virtual void compute_energies(const SGVector<float64_t> factor_data,
		SGVector<float64_t>& energies) const;

	/** Compute energy values from parameters for a specific factor.
	 *
	 * @param factor_data sparse factor data 
	 * @param energies forwarded energy table
	 */
	virtual void compute_energies(const SGSparseVector<float64_t> factor_data_sparse,
		SGVector<float64_t>& energies) const;

	/** compute parameter gradient from marginals and factor data
	 *
	 * @param factor_data dense factor data 
	 * @param marginals marginal distribution of the factor
	 * @param parameter_gradient gradient of factor parameters
	 * @param mult multiplier 
	 */
	virtual void compute_gradients(const SGVector<float64_t> factor_data,
		const SGVector<float64_t> marginals,
		SGVector<float64_t>& parameter_gradient, double mult) const;

	/** compute parameter gradient from marginals and factor data
	 *
	 * @param factor_data sparse factor data 
	 * @param marginals marginal distribution of the factor
	 * @param parameter_gradient gradient of factor parameters
	 * @param mult multiplier
	 */
	virtual void compute_gradients(const SGSparseVector<float64_t> factor_data_sparse,
		const SGVector<float64_t> marginals,
		SGVector<float64_t>& parameter_gradient, double mult) const;

};

}

#endif

