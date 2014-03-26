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

#include <stdio.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{

class CDistanceMachine;

/** @brief Class KNN, an implementation of the standard k-nearest neigbor
 * classifier.
 *
 * An example is classified to belong to the class of which the majority of the
 * k closest examples belong to. Formally, kNN is described as
 *
 * \f[
 *		label for x = \arg \max_{l} \sum_{i=1}^{k} [label of i-th example = l]
 * \f]
 *
 * This class provides a capability to do weighted classfication using:
 *
 * \f[
 *		label for x = \arg \max_{l} \sum_{i=1}^{k} [label of i-th example = l] q^{i},
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
		
		/** Defines the Mode of Operation for the KNN. There are 3 modes :
		  * 1) BruteForce.
		  * 2) CoverTree.
		  * 3) KDTree.
		  */
		enum EKNNMode { BruteForce, CoverTree, KDTree };

		/** default constructor */
		CKNN();

		/** constructor
		 *
		 * @param k k
		 * @param d distance
		 * @param trainlab labels for training
		 * @param mode Mode of Operation, default value = BruteForce
		 * @param leaf_size Leaf Size for KDTree, default value = 20
		 */
		CKNN(int32_t k, CDistance* d, CLabels* trainlab, EKNNMode mode = BruteForce,
			 int32_t leaf_size = 20);
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
		SGMatrix<int32_t> classify_for_multiple_k(CFeatures* data=NULL);

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

		/** set Mode of Operation for KNN
		 * @param mode
		 */
		inline void set_mode(EKNNMode mode) { m_mode = mode; }
		
		/** get Mode of Operation for KNN
		 * @return mode
		 */
		inline EKNNMode get_mode() const { return m_mode; }

		/** set Leaf Size of KDTree
		 * @param leaf_size
		 */
		inline void set_leaf_size(int32_t ls) { m_leaf_size = ls; }
		
		/** get Leaf Size of KDTree
		 * @return leaf_size
		 */
		inline int32_t get_leaf_size() const { return m_leaf_size; }
		
		/** @return object name */
		virtual const char* get_name() const { return "KNN"; }

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
	
	protected:
		/// the Mode of Operation
		EKNNMode m_mode;
		
		/// the k parameter in KNN
		int32_t m_k;

		/// parameter q of rank weighting
		float64_t m_q;

		///	number of classes (i.e. number of values labels can take)
		int32_t m_num_classes;

		///	smallest label, i.e. -1
		int32_t m_min_label;
		
		/// Leaf-Size, in case KDTree is used
		int32_t m_leaf_size;

		/** the actual trainlabels */
		SGVector<int32_t> m_train_labels;
};

}
#endif
