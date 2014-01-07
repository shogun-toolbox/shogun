/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Christian Widmer
 * Written (W) 2007-2010 Soeren Sonnenburg
 * Copyright (c) 2007-2009 The LIBLINEAR Project.
 * Copyright (C) 2007-2012 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBLINEARMTL_H___
#define _LIBLINEARMTL_H___

#include <lib/config.h>

#include <lib/common.h>
#include <base/Parameter.h>
#include <machine/LinearMachine.h>
#include <optimization/liblinear/shogun_liblinear.h>
#include <lib/SGSparseMatrix.h>

#include <map>

namespace shogun
{

#ifdef HAVE_LAPACK


/** @brief mapped sparse matrix for
 * representing graph relations of tasks
 */
class MappedSparseMatrix
{

    public:

    /** operator overload for matrix read only access
     * @param i_row
     * @param i_col
     */
    inline const float64_t operator()(index_t i_row, index_t i_col) const
    {

		// lookup complexity is O(log n)
		std::map<index_t, float64_t>::const_iterator it = data[i_row].find(i_col);

		if (it != data[i_row].end())
		{
			// use mapping for lookup
			return it->second;
		} else {
			return 0.0;
		}
	}

    /** set matrix from SGSparseMatrix
     * @param sgm
     */
    void set_from_sparse(const SGSparseMatrix<float64_t> &sgm)
    {
        data.clear();

        // deep copy sparse matrix
        for (int32_t i=0; i!=sgm.num_vectors; i++)
        {

            SGSparseVector<float64_t> ts_row = sgm.sparse_matrix[i];
            data.push_back(std::map<index_t, float64_t>());

            for (int32_t k=0; k!=ts_row.num_feat_entries; k++)
            {
				// get data from sparse matrix
				SGSparseVectorEntry<float64_t> e = ts_row.features[k];
                data[i][e.feat_index] = e.entry;
            }

        }
    }

	/** under-the-hood data structure  */
    std::vector< std::map<index_t, float64_t> > data;

};


/** @brief class to implement LibLinear */
class CLibLinearMTL : public CLinearMachine
{
	public:
		/** default constructor  */
		CLibLinearMTL();


		/** constructor (using L2R_L1LOSS_SVC_DUAL as default)
		 *
		 * @param C constant C
		 * @param traindat training features
		 * @param trainlab training labels
		 */
		CLibLinearMTL(
			float64_t C, CDotFeatures* traindat,
			CLabels* trainlab);

		/** destructor */
		virtual ~CLibLinearMTL();


		/** get classifier type
		 *
		 * @return the classifier type
		 */
		virtual EMachineType get_classifier_type() { return CT_LIBLINEAR; }

		/** set C
		 *
		 * @param c_neg C1
		 * @param c_pos C2
		 */
		inline void set_C(float64_t c_neg, float64_t c_pos) { C1=c_neg; C2=c_pos; }

		/** get C1
		 *
		 * @return C1
		 */
		inline float64_t get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline float64_t get_C2() { return C2; }

		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(float64_t eps) { epsilon=eps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline float64_t get_epsilon() { return epsilon; }

		/** set if bias shall be enabled
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** check if bias is enabled
		 *
		 * @return if bias is enabled
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** @return object name */
		virtual const char* get_name() const { return "LibLinearMTL"; }

		/** get the maximum number of iterations liblinear is allowed to do */
		inline int32_t get_max_iterations()
		{
			return max_iterations;
		}

		/** set the maximum number of iterations liblinear is allowed to do */
		inline void set_max_iterations(int32_t max_iter=1000)
		{
			max_iterations=max_iter;
		}

		/** set number of tasks */
		inline void set_num_tasks(int32_t nt)
		{
			num_tasks = nt;
		}

		/** set the linear term for qp */
		inline void set_linear_term(SGVector<float64_t> linear_term)
		{
			if (!m_labels)
				SG_ERROR("Please assign labels first!\n")

			int32_t num_labels=m_labels->get_num_labels();

			if (num_labels!=linear_term.vlen)
			{
				SG_ERROR("Number of labels (%d) does not match number"
						" of entries (%d) in linear term \n", num_labels,
						linear_term.vlen);
			}

			m_linear_term = linear_term;
		}

		/** set task indicator for lhs */
		inline void set_task_indicator_lhs(SGVector<int32_t> ti)
		{
			task_indicator_lhs = ti;
		}

		/** set task indicator for rhs */
		inline void set_task_indicator_rhs(SGVector<int32_t> ti)
		{
			task_indicator_rhs = ti;
		}

		/** set task similarity matrix */
		inline void set_task_similarity_matrix(SGSparseMatrix<float64_t> tsm)
		{
			task_similarity_matrix.set_from_sparse(tsm);
		}

		/** set graph laplacian */
		inline void set_graph_laplacian(SGMatrix<float64_t> lap)
		{
			graph_laplacian = lap;
		}

		/** get V
		 *
		 * @return matrix of weight vectors
		 */
		inline SGMatrix<float64_t> get_V()
		{
			return V;
		}

		/** get W
		 *
		 * @return matrix of weight vectors
		 */
		inline SGMatrix<float64_t> get_W()
		{

            int32_t w_size = V.num_rows;

            SGMatrix<float64_t> W = SGMatrix<float64_t>(w_size, num_tasks);
            for(int32_t k=0; k<w_size*num_tasks; k++)
            {
                W.matrix[k] = 0;
            }

            for (int32_t s=0; s<num_tasks; s++)
            {
                float64_t* v_s = V.get_column_vector(s);
                for (int32_t t=0; t<num_tasks; t++)
                {
                    float64_t sim_ts = task_similarity_matrix(s,t);
                    for(int32_t i=0; i<w_size; i++)
                    {
                        W.matrix[t*w_size + i] += sim_ts * v_s[i];
                    }
                }
            }

			return W;
		}

		/** get alphas
		 *
		 * @return matrix of example weights alphas
		 */
		inline SGVector<float64_t> get_alphas()
		{
			return alphas;
		}

		/** compute primal objective
		 *
		 * @return primal objective
		 */
		virtual float64_t compute_primal_obj();

		/** compute dual objective
		 *
		 * @return dual objective
		 */
		virtual float64_t compute_dual_obj();

		/** compute duality gap
		 *
		 * @return duality gap
		 */
		virtual float64_t compute_duality_gap();


	protected:
		/** train linear SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	private:
		/** set up parameters */
        void init();

		void solve_l2r_l1l2_svc(
			const liblinear_problem *prob, double eps, double Cp, double Cn);


	protected:
		/** C1 */
		float64_t C1;
		/** C2 */
		float64_t C2;
		/** if bias shall be used */
		bool use_bias;
		/** epsilon */
		float64_t epsilon;
		/** maximum number of iterations */
		int32_t max_iterations;

		/** precomputed linear term */
		SGVector<float64_t> m_linear_term;

		/** keep track of alphas */
		SGVector<float64_t> alphas;

		/** set number of tasks */
        int32_t num_tasks;

		/** task indicator left hand side */
		SGVector<int32_t> task_indicator_lhs;

		/** task indicator right hand side */
		SGVector<int32_t> task_indicator_rhs;

		/** task similarity matrix */
		//SGMatrix<float64_t> task_similarity_matrix;
		//SGSparseMatrix<float64_t> task_similarity_matrix;
		MappedSparseMatrix task_similarity_matrix;

		/** task similarity matrix */
		SGMatrix<float64_t> graph_laplacian;

		/** parameter matrix n * d */
		SGMatrix<float64_t> V;

        /** duality gap */
        float64_t duality_gap;

};

#endif //HAVE_LAPACK

} /* namespace shogun  */

#endif //_LIBLINEARMTL_H___
