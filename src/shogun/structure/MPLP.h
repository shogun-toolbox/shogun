/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Copyright (C) 2014 Jiaolong Xu
 */

#ifndef __MPLP_H__
#define __MPLP_H__

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/Factor.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/structure/MDArray.h>

#include <map>

namespace shogun
{
#define IGNORE_IN_CLASSLIST

/** "Region" simply corresponds to an algorithmic concept, specifying both
 * dual variables shared between a larger intersection set and smaller ones,
 * and an update strategy. Every region has a corresponding intersection set,
 * and all potentials are inside of the intersection sets. The objective function
 * can be computed simply by summing over the intersection sets.
 * See Sontag's Ph.D. thesis page 104.
 */
IGNORE_IN_CLASSLIST class CRegion
{
public:
	/** Constructor
	 * @param region_inds the variables in the region
	 * @param all_intersects all intersects in the region, each intersect contains a group of variables
	 * @param intersect_inds the indeces of the intersetion sets
	 * @param var_sizes the cardinalities of each variable
	 * @param region_intersect every region has a corresponding intersetion set, the index of this set*/
	CRegion(vector<int32_t> region_inds, vector<vector<int32_t> > all_intersects, vector<int32_t> intersect_inds, vector<int32_t> var_sizes, int32_t region_intersect);

	/** Update messages
	 *
	 * This algorithm is a modification of the original MPLP algorithm,
	 * described in Figure A-1 on Sontag's Ph.D. thesis (page 104).
	 */
	void update_messages(vector<CMDArray> &sum_into_intersects);

	/** Maximization into multiple subsets
	 *
	 * @param big_arr current set
	 * @param all_subset_inds indecise of all subsets
	 * @param all_max_res results after maximization*/
	void max_into_multiple_subsets(CMDArray big_arr, vector<vector<int32_t> > &all_subset_inds, vector<CMDArray> &all_max_res) const;

	/** Get the number of the variables*/
	int32_t get_num_vars()
	{
		return m_var_sizes.size();
	};

public:
	vector<int32_t> m_region_inds; // the variables in the region
	vector<int32_t> m_intersect_inds; // specifies the indices of the intersection sets

	// specifies the indices corresponding to the position of each intersection set in the region
	vector<vector<int32_t> > m_inds_of_intersects;

	int32_t m_region_intersect; // every region has a corresponding intersection set. The index of this set.
	vector<CMDArray> m_msgs_from_region; // contains the messages from each region to its intersection sets
	vector<int32_t> m_var_sizes; // the size (size of label space) of each variable
};

/** MPLP (Max-product LP Relaxation) inference for fatcor graph
 *
 * Please refer to following paper for more details:
 *
 * Approximate Inference in Graphical Models using LP Relaxations.
 * David Sontag
 * Ph.D. thesis, Massachusetts Institute of Technology, 2010.
 *
 * Tightening LP Relaxations for MAP using Message Passing
 * David Sontag, Talya Meltzer, Amir Globerson, Tommi Jaakkola and Yair Weiss
 * Uncertainty in Artificial Intelligence (UAI). Helsinki, Finland. 2008.
 *
 * Fixing max-product: Convergent message passing algorithms for MAP LP-relaxations
 * Amir Globerson, Tommi Jaakkola
 * Advances in Neural Information Processing Systems (NIPS) 21. Vancouver, Canada. 2007.
 */
IGNORE_IN_CLASSLIST class CMPLP: public CMAPInferImpl
{
public:
	/** Parameter for MPLP */
	struct Parameter
	{
		Parameter(const int32_t max_iter = 1000, const int32_t max_iter_inner = 16,
		          const float64_t obj_del_thr = 0.0002, const float64_t int_gap_thr = 0.0002)
			: m_max_iter(max_iter),
			  m_max_iter_inner(max_iter_inner),
			  m_obj_del_thr(obj_del_thr),
			  m_int_gap_thr(int_gap_thr)
		{}

		int32_t m_max_iter; // maximum number of iterations for the initial LP
		int32_t m_max_iter_inner; // maximum number of iteration for inner loop of LP when delta objective value is under threshold
		float64_t m_obj_del_thr; // threshold of the delta objective value
		float64_t m_int_gap_thr; // threshold of objective gap betweeb current and best integer assignment
	};

public:
	/** Constructor */
	CMPLP();

	/** Constructor
	 *
	 * @param fg factor graph
	 */
	CMPLP(CFactorGraph* fg, Parameter param = Parameter());

	/** Destructor */
	virtual ~CMPLP();

	/** @return class name */
	virtual const char* get_name() const
	{
		return "MPLP";
	}

	/** Inference
	 *
	 * @param the assignment
	 * @return the total energy after doing inference
	 */
	virtual float64_t inference(SGVector<int32_t> assignment);

private:
	/** Initialize MPLP with factor graph */
	void init();

	/** Initialize MPLP
	 *
	 * @param var_sizes size of each variable
	 * @param all_region_inds contains the variables in each region
	 * @param all_lambdas contains the energy table in each region
	 */
	void init(vector<int32_t> var_sizes, vector<vector<int32_t> > all_region_inds, vector<vector<float64_t> > all_lambdas);

	/** Main logic of MPLP
	 *
	 * @param niter number of iteration
	 * @param object_del_thr threshold of the objective value gap between current and last iteration
	 * @param int_gap_thr threshold of the objective value gap between current and best integer assignment
	 */
	void run_mplp(int32_t niter, float64_t obj_del_thr, float64_t int_gap_thr);

	/** Compute the objective value with respective to the current integer assignment
	 *
	 * @param assignment integer assignment to the varibles
	 * @return the objective value with respect to the integer assignment
	 */
	float64_t compute_int_val(vector<int32_t> &assignment) const;

	/** single node decoding */
	float64_t local_decode();

	/** Check current integer assignment and update the result
	 *
	 * @return primal objective of this mplp instance
	 */
	float64_t update_result();

public:
	Parameter m_param; // MPLP parameter
	float64_t m_best_val; // the best primal objective so far
	float64_t m_last_obj; // last primal objective value
	float64_t m_obj_del; // delta objective value, i.e., current - last

	vector<vector<int32_t> > m_all_intersects; // all intersections of the factor graph
	vector<CRegion> m_all_regions; // all regions in the factor graph
	vector<CMDArray> m_sum_into_intersects; //
	vector<int32_t> m_var_sizes; // the size of the variables
	vector<int32_t> m_decoded_res; // integer assignment
	vector<int32_t> m_best_decoded_res; // best assignment so far
	vector<CMDArray> m_single_node_lambdas; // i.e., unary potential
	vector<CMDArray> m_region_lambdas; // energy of each region

	// This map allows us to quickly look up the index of edge intersection sets
	map<pair<int32_t, int32_t>, int32_t> m_intersect_map;
};
}

#endif
