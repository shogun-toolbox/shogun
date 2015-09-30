/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef RELAXEDTREE_H__
#define RELAXEDTREE_H__

#include <utility>
#include <vector>

#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/RelaxedTreeNodeData.h>

namespace shogun
{

class CBaseMulticlassMachine;

/** RelaxedTree refer to a tree-style multiclass classifier proposed in
 * the following paper.
 *
 *   Tianshi Gao and Daphne Koller. Discriminative Learning of Relaxed
 *   Hierarchy for Large-scale Visual Recognition. In IEEE International
 *   Conference on Computer Vision (ICCV), 2011. (Oral presentation)
 */
class CRelaxedTree: public CTreeMachine<RelaxedTreeNodeData>
{
public:
	/** constructor */
	CRelaxedTree();

	/** destructor */
	virtual ~CRelaxedTree();

	/** get name */
	virtual const char* get_name() const { return "RelaxedTree"; }

	/** apply machine to data in means of multiclass classification problem */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

	/** set features
	 * @param feats features
	 */
	void set_features(CDenseFeatures<float64_t> *feats)
	{
		SG_REF(feats);
		SG_UNREF(m_feats);
		m_feats = feats;
	}

	/** set kernel
	 * @param kernel the kernel to be used
	 */
	virtual void set_kernel(CKernel *kernel)
	{
		SG_REF(kernel);
		SG_UNREF(m_kernel);
		m_kernel = kernel;
	}

	/** set labels
	 *
	 * @param lab labels
	 */
	virtual void set_labels(CLabels* lab)
	{
		CMulticlassLabels *mlab = dynamic_cast<CMulticlassLabels *>(lab);
		REQUIRE(lab, "requires MulticlassLabes\n")

		CMachine::set_labels(mlab);
		m_num_classes = mlab->get_num_classes();
	}

	/** set machine for confusion matrix
	 * @param machine the multiclass machine for initializing the confusion matrix
	 */
	void set_machine_for_confusion_matrix(CBaseMulticlassMachine *machine)
	{
		SG_REF(machine);
		SG_UNREF(m_machine_for_confusion_matrix);
		m_machine_for_confusion_matrix = machine;
	}

	/** set SVM C: parameter for relax variables. See eq (1) in the paper.
	 * @param C svm C
	 */
	void set_svm_C(float64_t C)
	{
		m_svm_C = C;
	}
	/** get SVM C
	 * @return svm C
	 */
	float64_t get_svm_C() const
	{
		return m_svm_C;
	}

	/** set SVM epsilon
	 * @param epsilon SVM epsilon
	 */
	void set_svm_epsilon(float64_t epsilon)
	{
		m_svm_epsilon = epsilon;
	}
	/** get SVM epsilon
	 * @return svm epsilon
	 */
	float64_t get_svm_epsilon() const
	{
		return m_svm_epsilon;
	}

	/** set parameter A for controlling the trade-off of encouraging more classes
	 * to participating the discriminating at each level (i.e. not be ignored). See
	 * eq (1) in the paper.
	 * @param A
	 */
	void set_A(float64_t A)
	{
		m_A = A;
	}
	/** get parameter A
	 * @return A
	 */
	float64_t get_A() const
	{
		return m_A;
	}

	/** set parameter B for constraining the inbalance of binary colorization. See
	 * eq (1) in the paper.
	 * @param B
	 */
	void set_B(int32_t B)
	{
		m_B = B;
	}
	/** get parameter B
	 * @return B
	 */
	int32_t get_B() const
	{
		return m_B;
	}

	/** set max number of iteration in alternating optimization
	 * @param n_iter number of iterations
	 */
	void set_max_num_iter(int32_t n_iter)
	{
		m_max_num_iter = n_iter;
	}
	/** get max number of iteration in alternating optimization
	 * @return number of iterations
	 */
	int32_t get_max_num_iter() const
	{
		return m_max_num_iter;
	}

	/** train machine
	 *
	 * @param data training data (parameter can be avoided if distance or
	 * kernel-based classifiers are used and distance/kernels are
	 * initialized with train data).
	 * If flag is set, model features will be stored after training.
	 *
	 * @return whether training was successful
	 */
	virtual bool train(CFeatures* data=NULL)
	{
		return CMachine::train(data);
	}

	/** entry type */
	typedef std::pair<std::pair<int32_t, int32_t>, float64_t> entry_t;
protected:
	/** apply to one instance.
	 *
	 * Note this method is not made public so that not be called from
	 * external source. This is because preparation have to be done
	 * before calling this (mainly setup the kernel for submachines).
	 */
	float64_t apply_one(int32_t idx);

	/** train machine
	 *
	 * @param data training data
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(CFeatures* data);

	/** train node */
	bnode_t *train_node(const SGMatrix<float64_t> &conf_mat, SGVector<int32_t> classes);
	/** init node */
	std::vector<entry_t> init_node(const SGMatrix<float64_t> &global_conf_mat, SGVector<int32_t> classes);
	/** train node with initialization */
	SGVector<int32_t> train_node_with_initialization(const CRelaxedTree::entry_t &mu_entry, SGVector<int32_t> classes, CSVM *svm);

	/** compute score */
	float64_t compute_score(SGVector<int32_t> mu, CSVM *svm);
	/** color label space */
	SGVector<int32_t> color_label_space(CSVM *svm, SGVector<int32_t> classes);
	/** evaluate binary model K */
	SGVector<float64_t> eval_binary_model_K(CSVM *svm);

	/** enforce balance constraints upper */
	void enforce_balance_constraints_upper(SGVector<int32_t> &mu, SGVector<float64_t> &delta_neg, SGVector<float64_t> &delta_pos, int32_t B_prime, SGVector<float64_t>& xi_neg_class);
	/** enforce balance constraints lower */
	void enforce_balance_constraints_lower(SGVector<int32_t> &mu, SGVector<float64_t> &delta_neg, SGVector<float64_t> &delta_pos, int32_t B_prime, SGVector<float64_t>& xi_neg_class);

	/** maximum number of iterations */
	int32_t m_max_num_iter;
	/** A */
	float64_t m_A;
	/** B */
	int32_t m_B;
	/** svm C */
	float64_t m_svm_C;
	/** svm epsilon */
	float64_t m_svm_epsilon;
	/** kernel */
	CKernel *m_kernel;
	/** features */
	CDenseFeatures<float64_t> *m_feats;
	/** machine for confusion matrix computation */
	CBaseMulticlassMachine *m_machine_for_confusion_matrix;
	/** number of classes */
	int32_t m_num_classes;
};

} /* shogun */

#endif /* end of include guard: RELAXEDTREE_H__ */

