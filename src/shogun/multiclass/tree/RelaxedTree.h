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

#include <shogun/features/DenseFeatures.h>
#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/RelaxedTreeNodeData.h>

namespace shogun
{

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

	/** set labels
	 *
	 * @param lab labels
	 */
	virtual void set_labels(CLabels* lab)
	{
		CMulticlassLabels *mlab = dynamic_cast<CMulticlassLabels *>(lab);
		if (lab == NULL)
			SG_ERROR("requires MulticlassLabes\n");

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

	typedef std::pair<std::pair<int32_t, int32_t>, float64_t> entry_t;
protected:
	/** train machine
	 *
	 * @param data training data 
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(CFeatures* data);

	void train_node(const SGMatrix<float64_t> &conf_mat, SGVector<int32_t> classes);
	std::vector<entry_t> init_node(const SGMatrix<float64_t> &global_conf_mat, SGVector<int32_t> classes);

	CDenseFeatures<float64_t> *m_feats;
	CBaseMulticlassMachine *m_machine_for_confusion_matrix;
	int32_t m_num_classes;
};

} /* shogun */ 

#endif /* end of include guard: RELAXEDTREE_H__ */

