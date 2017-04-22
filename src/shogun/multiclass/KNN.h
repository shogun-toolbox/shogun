/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2011 Sergey Lisitsyn
 * Written (W) 2012 Fernando José Iglesias García, cover tree support
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef _KNN_H__
#define _KNN_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>
#include <shogun/multiclass/KNNSolver.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/BruteKNNSolver.h>
#include <shogun/multiclass/KDTreeKNNSolver.h>
#include <shogun/multiclass/CoverTreeKNNSolver.h>
#include <shogun/multiclass/LSHKNNSolver.h>

namespace shogun
{
	enum KNN_SOLVER
	{
		KNN_BRUTE,
		KNN_KDTREE,
		KNN_COVER_TREE,
#ifdef HAVE_CXX11
		KNN_LSH
#endif
	};

class CDistanceMachine;

/** @brief Class KNN, an implementation of the standard k-nearest neigbor
 * classifier.
 *
 * An example is classified to belong to the class of which the majority of the
 * k closest examples belong to. Formally, kNN is described as
 *
 * \f[
 *		y_{x} = \arg \max_{l} \sum_{i=1}^{k} I[y_{i} = l],
 * \f]
 * 
 * where \f$y_{m}\f$ denotes the label of the \f$m^{th}\f$ example, and the
 * indicator function \f$I[a = b]\f$ equals 1 if a = b and zero otherwise.
 *
 * This class provides a capability to do weighted classfication using:
 *
 * \f[
 *		y_{x} = \arg \max_{l} \sum_{i=1}^{k} I[y_{i} = l] q^{i},
 * \f]
 *
 * where \f$|q|<1\f$.
 *
 * To avoid ties, k should be an odd number. To define how close examples are
 * k-NN requires a CDistance object to work with (e.g., CEuclideanDistance ).
 *
 * Note that k-NN has zero training time but classification times increase
 * dramatically with the number of examples. Also note that k-NN is capable of
 * multi-class-classification. And finally, in case of k=1 classification will
 * take less time with an special optimization provided.
 */
class CKNN : public CDistanceMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** default constructor */
		CKNN();

		/** constructor
		 *
		 * @param k k
		 * @param d distance
		 * @param trainlab labels for training
		 */
		CKNN(int32_t k, CDistance* d, CLabels* trainlab, KNN_SOLVER knn_solver=KNN_BRUTE);

		virtual ~CKNN();

		/** get classifier type
		 *
		 * @return classifier type KNN
		 */
		virtual EMachineType get_classifier_type() { return CT_KNN; }

		/**
		 * for each example in the rhs features of the distance member, find the m_k
		 * nearest neighbors among the vectors in the lhs features
		 *
		 * @return matrix with indices to the nearest neighbors, the dimensions of the
		 * matrix are k rows and n columns, where n is the number of feature vectors in rhs;
		 * among the nearest neighbors, the closest are in the first row, and the furthest
		 * in the last one
		 */
		SGMatrix<index_t> nearest_neighbors();

		/** classify objects
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

		/// get output for example "vec_idx"
		virtual float64_t apply_one(int32_t vec_idx)
		{
			SG_ERROR("for performance reasons use apply() instead of apply(int32_t vec_idx)\n")
			return 0;
		}

		/** classify all examples for 1...k
		 *
		 */
		SGMatrix<int32_t> classify_for_multiple_k();

		/** load from file
		 *
		 * @param srcfile file to load from
		 * @return if loading was successful
		 */
		virtual bool load(FILE* srcfile);

		/** save to file
		 *
		 * @param dstfile file to save to
		 * @return if saving was successful
		 */
		virtual bool save(FILE* dstfile);

		/** set k
		 *
		 * @param k k to be set
		 */
		inline void set_k(int32_t k)
		{
			ASSERT(k>0)
			m_k=k;
		}

		/** get k
		 *
		 * @return value of k
		 */
		inline int32_t get_k()
		{
			return m_k;
		}

		/** set q
		 * @param q value
		 */
		inline void set_q(float64_t q)
		{
			ASSERT(q<=1.0 && q>0.0)
			m_q = q;
		}

		/** get q
		 * @return q parameter
		 */
		inline float64_t get_q() { return m_q; }

		/** get leaf size for KD-Tree
		 *	@return leaf_size
		 */
		inline int32_t get_leaf_size() const {return m_leaf_size; }

		/** Set leaf size for KD-Tree 
		 *	@param leaf_size
		 */
		inline void set_leaf_size(int32_t leaf_size)
		{
			m_leaf_size = leaf_size;
		}

		/** @return object name */
		virtual const char* get_name() const { return "KNN"; }

		/**
		 * @return the currently used KNN algorithm
		 */
		inline KNN_SOLVER get_knn_solver_type()
		{
			return m_knn_solver;
		}

		/** set the KNN algorithm
		 *
		 * @param knn_solver Used solver
		 */
		inline void set_knn_solver_type(KNN_SOLVER knn_solver)
		{
			m_knn_solver = knn_solver;
		}

#ifdef HAVE_CXX11
		/** set parameters for LSH solver
		  * @param l number of hash tables for LSH
		  * @param t number of probes per query for LSH
		  */
		inline void set_lsh_parameters(int32_t l, int32_t t)
		{
			m_lsh_l = l;
			m_lsh_t = t;
		}
#endif

	protected:
		/** Stores feature data of underlying model.
		 *
		 * Replaces lhs and rhs of underlying distance with copies of themselves
		 */
		virtual void store_model_features();

		/** classify all examples with nearest neighbor (k=1)
		 * @return classified labels
		 */
		virtual CMulticlassLabels* classify_NN();

		/** init distances to test examples
		 * @param data test examples
		 */
		void init_distance(CFeatures* data);

		/** train k-NN classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	private:
		void init();

		/** compute the histogram of class outputs of the k nearest
		 *  neighbors to a test vector and return the index of the most
		 *  frequent class
		 *
		 * @param classes vector used to store the histogram
		 * @param train_lab class indices of the training data. If the cover
		 * tree is not used, the elements are ordered by increasing distance
		 * and there are elements for each of the training vectors. If the cover
		 * tree is used, it contains just m_k elements not necessary ordered.
		 *
		 * @return index of the most frequent class, class detected by KNN
		 */
		int32_t choose_class(float64_t* classes, int32_t* train_lab);

		/** compute the histogram of class outputs of the k nearest neighbors
		 *  to a test vector, using k from 1 to m_k, and write the most frequent
		 *  class for each value of k in output, using a distance equal to step
		 *  between elements in the output array
		 *
		 * @param output return value where the most frequent classes are written
		 * @param classes vector used to store the histogram
		 * @param train_lab class indices of the training data; no matter the cover tree
		 * is used or not, the neighbors are ordered by distance to the test vector
		 * in ascending order
		 * @param step distance between elements to be written in output
		 */
		void choose_class_for_multiple_k(int32_t* output, int32_t* classes, int32_t* train_lab, int32_t step);

		/** 
		 * To init the solver pointer indicated which solver will been used to classify_objects
		 */
		void init_solver(KNN_SOLVER knn_solver);

	protected:
		/// the k parameter in KNN
		int32_t m_k;

		/// parameter q of rank weighting
		float64_t m_q;

		/// number of classes (i.e. number of values labels can take)
		int32_t m_num_classes;

		/// smallest label, i.e. -1
		int32_t m_min_label;

		/** the actual trainlabels */
		SGVector<int32_t> m_train_labels;

		/// Solver for KNN
		CKNNSolver* solver;

		KNN_SOLVER m_knn_solver;

		int32_t m_leaf_size;

#ifdef HAVE_CXX11
		/* Number of hash tables for LSH */
		int32_t m_lsh_l;

		/* Number of probes per query for LSH */
		int32_t m_lsh_t;
#endif
};

}
#endif
