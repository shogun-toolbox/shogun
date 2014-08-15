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
#include <shogun/lib/SGNDArray.h>

#include <vector>

using namespace std;

namespace shogun
{
#define IGNORE_IN_CLASSLIST

/** Edge factor
 *
 * Edge factor represents an edge in EMPLP (edge max-product linear programming).
 * EMPLP applies block coordinate desent in the LP relaxation dual. It is equivalent
 * to fix some edges and minimizing over non-fixed edges at each iteration. The minimization
 * here is done by updating the edge messages.
 *
 * Edge factor can also be used for Generilized EMPLP, i.e., GEMPLP. In this case, edge factor
 * represents a cluster of variables. For details, please refer to the following publication:
 *
 * [1] Fixing max-product: Convergent message passing algorithms for MAP LP-relaxations,
 * Amir Globerson, Tommi Jaakkola,
 * Advances in Neural Information Processing Systems (NIPS). Vancouver, Canada. 2007.
 */
IGNORE_IN_CLASSLIST class CEdgeFactor : public CFactor
{
public:

	/** Constructor
	 *
	 * @param factor the general factor in the factor graph
	 * @param ind_factor the index of this factor in all factors.
	 *        It is also the index of current variable set in all variable sets
	 */
	CEdgeFactor(CFactor* factor, int32_t ind_factor);

	/** @return class name */
	virtual const char* get_name() const
	{
		return "EdgeFactor";
	}

	/** Update messages
	 *
	 * Please refer to Figure 1 "EMPLP" in NIPS paper of
	 * A. Globerson and T. Jaakkola [1] for more details.
	 *
	 * @param msgs_all_factors messages of all factors
	 */
	void update_messages(vector< SGNDArray<float64_t> > &msgs_all_factors);

	/** Maximization into multiple subsets
	 *
	 * @param big_arr current set
	 * @param all_subset_inds indecise of all subsets
	 * @param all_max_res results after maximization
	 * */
	void max_into_multiple_subsets(SGNDArray<float64_t> big_arr,
	                               vector<vector<int32_t> > &all_subset_inds,
	                               vector< SGNDArray<float64_t> > &all_max_res) const;

public:
	/** specifies the indices corresponding to the
	 position of each variable set on the edge */
	vector<vector<int32_t> > m_inds_on_edge;
	/** the index of this factor in the gloabel factors. */
	int32_t m_ind_factor;
	/** contains the messages from each edge to its variable sets */
	vector<SGNDArray<float64_t> > m_msgs_from_edge;
};

/** MPLP (Max-product LP Relaxation) inference for fatcor graph
 *
 * Please refer to following paper for more details:
 *
 * [1] Fixing max-product: Convergent message passing algorithms for MAP LP-relaxations
 * Amir Globerson, Tommi Jaakkola
 * Advances in Neural Information Processing Systems (NIPS). Vancouver, Canada. 2007.
 *
 * [2] Approximate Inference in Graphical Models using LP Relaxations.
 * David Sontag
 * Ph.D. thesis, Massachusetts Institute of Technology, 2010.
 *
 * The original implementation of MPLP can be found:
 * http://cs.nyu.edu/~dsontag/code/mplp_ver2.tgz
 * http://cs.nyu.edu/~dsontag/code/mplp_ver1.tgz
 */
IGNORE_IN_CLASSLIST class CMPLP: public CMAPInferImpl
{
public:
	/** Parameter for MPLP */
	struct Parameter
	{
		Parameter(const int32_t max_iter = 1000,
		          const int32_t max_iter_inner = 16,
		          const float64_t obj_del_thr = 0.0002,
		          const float64_t int_gap_thr = 0.0002)
			: m_max_iter(max_iter),
			  m_max_iter_inner(max_iter_inner),
			  m_obj_del_thr(obj_del_thr),
			  m_int_gap_thr(int_gap_thr)
		{}

		/** maximum number of outer iterations*/
		int32_t m_max_iter;
		/** maximum number of iteration for inner loop of LP */
		int32_t m_max_iter_inner;
		/** threshold of the delta objective value */
		float64_t m_obj_del_thr;
		/** threshold of objective gap betweeb current and best integer assignment */
		float64_t m_int_gap_thr;
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

	/** Main logic of MPLP

	 * Please refer to Figure 1 "EMPLP" in NIPS paper of
	 * A. Globerson and T. Jaakkola [1] for more details.
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
	float64_t evaluate_assignment(SGVector<int32_t> &assignment) const;

	/** get the current assignment and compute the objective value */
	float64_t get_assignment();

public:
	/** MPLP parameter */
	Parameter m_param;
	/** the best primal objective so far */
	float64_t m_best_val;
	/** last primal objective value */
	float64_t m_last_obj;
	/** delta objective value, i.e., current - last */
	float64_t m_obj_del;

	/** all intersections of the factor graph */
	vector<vector<int32_t> > m_all_varSets;
	/** all edge factors in the factor graph */
	vector<CEdgeFactor> m_all_edges;
	/** store the messages of all factors */
	vector<SGNDArray<float64_t> > m_msgs_all_factors;
	/** current assignment */
	SGVector<int32_t> m_assignment;
	/** best assignment so far */
	SGVector<int32_t> m_best_assignment;
};
}
#endif
