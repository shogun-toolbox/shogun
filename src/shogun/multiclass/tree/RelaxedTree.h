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
protected:
	/** train machine
	 *
	 * @param data training data 
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(CFeatures* data);

	CDenseFeatures<float64_t> *m_feats;
};

} /* shogun */ 

#endif /* end of include guard: RELAXEDTREE_H__ */

