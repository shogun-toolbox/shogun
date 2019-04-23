/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shell Hu, Yuyu Zhang, Bjoern Esser
 */

#ifndef __FACTORGRAPH_H__
#define __FACTORGRAPH_H__

#include <shogun/lib/config.h>

#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/Factor.h>
#include <shogun/labels/FactorGraphLabels.h>
#include <shogun/structure/DisjointSet.h>

namespace shogun
{

/** @brief Class FactorGraph a factor graph is a structured input in general
 */
class FactorGraph : public SGObject
{

public:
	FactorGraph();

	/** Constructor
	 *
	 * @param card cardinalities of all the variables in the factor graph
	 */
	FactorGraph(const SGVector<int32_t> card);

	/** Copy constructor
	 *
	 * @param fg a factor graph instance
	 */
	FactorGraph(const FactorGraph &fg);

	/** deconstructor */
	~FactorGraph();

	/** @return class name */
	virtual const char* get_name() const { return "FactorGraph"; }

	/** add a factor
	 *
	 * @param factor a factor pointer
	 */
	void add_factor(std::shared_ptr<Factor> factor);

	/** add a data source
	 *
	 * @param datasource a factor data source
	 */
	void add_data_source(std::shared_ptr<FactorDataSource> datasource);

	/** @return all the factors */
	std::shared_ptr<DynamicObjectArray> get_factors() const;

	/** @return all the shared data */
	std::shared_ptr<DynamicObjectArray> get_factor_data_sources() const;

	/** @return number of factors */
	int32_t get_num_factors() const;

	/** @return cardinalities */
	SGVector<int32_t> get_cardinalities() const;

	/** set cardinalities
	 *
	 * @param cards cardinalities of all variables
	 */
	void set_cardinalities(SGVector<int32_t> cards);

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
	float64_t evaluate_energy(std::shared_ptr<const FactorGraphObservation> obs) const;

	/** @return energy table for the graph */
	SGVector<float64_t> evaluate_energies() const;

	/** @return disjoint set */
	std::shared_ptr<DisjointSet> get_disjoint_set() const;

	/** @return number of edges */
	int32_t get_num_edges() const;

	/** @return number of variable nodes */
	int32_t get_num_vars() const;

	/** connect graph nodes by performing union-find algorithm
	 * NOTE: make sure call this func before performing inference
	 * and other functions related to structure analysis
	 */
	void connect_components();

	/** @return is acyclic graph or not */
	bool is_acyclic_graph() const;

	/** @return is connected graph or not */
	bool is_connected_graph() const;

	/** @return is tree graph or not */
	bool is_tree_graph() const;

	/** perform loss-augmentation
	 *
	 * @param gt an observation (states and loss weights are stored in it)
	 */
	virtual void loss_augmentation(std::shared_ptr<FactorGraphObservation> gt);

	/** perform loss-augmentation
	 *
	 * @param states_gt ground truth states
	 * @param loss weighted loss for each variable
	 */
	virtual void loss_augmentation(SGVector<int32_t> states_gt, \
		SGVector<float64_t> loss = SGVector<float64_t>());

private:
	/** register parameters */
	void register_parameters();

	/** initialization */
	void init();

protected:
	// TODO: FactorNode, VariableNode, such that they have IDs

	/** cardinalities */
	SGVector<int32_t> m_cardinalities;

	/** added factors */
	std::shared_ptr<DynamicObjectArray> m_factors;

	/** added data sources */
	std::shared_ptr<DynamicObjectArray> m_datasources;

	/** disjoint set */
	std::shared_ptr<DisjointSet> m_dset;

	/** if has circle in the graph */
	bool m_has_cycle;

	/** number of edges */
	int32_t m_num_edges;

};

}

#endif

