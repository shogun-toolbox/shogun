/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Copyright (C) 2014 Jiaolong Xu
 */

#ifndef __GEMPLP_H__
#define __GEMPLP_H__

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/Factor.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/lib/SGNDArray.h>

#include <vector>
#include <set>

using namespace std;

namespace shogun
{
/** GEMPLP (Generalized Max-product LP Relaxation) inference for factor graph
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
 * The original implementation of GEMPLP can be found:
 * http://cs.nyu.edu/~dsontag/code/mplp_ver2.tgz
 * http://cs.nyu.edu/~dsontag/code/mplp_ver1.tgz
 */
class CGEMPLP: public CMAPInferImpl
{
public:
	/** Parameter for GEMPLP */
	struct Parameter
	{
		Parameter(const int32_t max_iter = 1000,
		          const float64_t obj_del_thr = 0.0002,
		          const float64_t int_gap_thr = 0.0002)
			: m_max_iter(max_iter),
			  m_obj_del_thr(obj_del_thr),
			  m_int_gap_thr(int_gap_thr)
		{}

		/** maximum number of outer iterations*/
		int32_t m_max_iter;
		/** threshold of the delta objective value, i.e., last_obj - curr_obj */
		float64_t m_obj_del_thr;
		/** threshold of the duality gap, i.e., primal_obj - dual_obj */
		float64_t m_int_gap_thr;
	};

public:
	/** Constructor */
	CGEMPLP();

	/** Constructor
	 *
	 * @param fg factor graph
	 * @param param parameters
	 */
	CGEMPLP(CFactorGraph* fg, Parameter param = Parameter());

	/** Destructor */
	virtual ~CGEMPLP();

	/** @return class name */
	virtual const char* get_name() const
	{
		return "GEMPLP";
	}

	/** Inference
	 *
	 * @param assignment the assignment
	 * @return the total energy after doing inference
	 */
	virtual float64_t inference(SGVector<int32_t> assignment);

private:
	/** Initialize GEMPLP with factor graph */
	void init();

	/** Message updating on a region
	 *
	 * Please refer to "GEMPLP" in NIPS paper of
	 * A. Globerson and T. Jaakkola [1] for more details.
	 *
	 */
	void update_messages(int32_t id_region);

public:
	/** Computer the maximum value along the sub-dimension
	 *
	 * @param tar_arr target nd array
	 * @param subset_inds sub-dimension indices
	 * @param max_res the result nd array
	 */
	void max_in_subdimension(SGNDArray<float64_t> tar_arr, SGVector<int32_t> &subset_inds, SGNDArray<float64_t> &max_res) const;

	/** Find intersection index between regions
	 *
	 * @param region_A region A
	 * @param region_B region B
	 * @return index in all intersection
	 */
	int32_t find_intersection_index(SGVector<int32_t> region_A, SGVector<int32_t> region_B);

	/** Convert original energies to potentials of the region
	 * GEMPLP objective function is a maximazation function, we use - energy
	 * as potential. The indices of the table factor energy also need to be
	 * converted to the order of the nd array.
	 *
	 * @param factor factor which contains energy
	 * @return potential of the region in MPLP
	 */
	SGNDArray<float64_t> convert_energy_to_potential(CFactor* factor);

public:
	/** GEMPLP parameter */
	Parameter m_param;
	/** all factors in the graph*/
	CDynamicObjectArray* m_factors;
	/** all intersections */
	vector<SGVector<int32_t> > m_all_intersections;
	/** the intersection indices (node indices) on each region */
	vector<set<int32_t> > m_region_intersections;
	/** the indices (orders in the region) of the intersections on each region */
	vector<vector<SGVector<int32_t> > > m_region_inds_intersections;
	/** store the sum of messages into intersections */
	vector<SGNDArray<float64_t> > m_msgs_into_intersections;
	/** store the messages from region to intersection */
	vector<vector<SGNDArray<float64_t> > > m_msgs_from_region;
	/** store the original (-) energy of the factors */
	vector<SGNDArray<float64_t> > m_theta_region;
};
}
#endif
