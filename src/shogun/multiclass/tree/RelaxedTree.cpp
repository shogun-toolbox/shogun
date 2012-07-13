/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <limits>
#include <algorithm>

#include <shogun/multiclass/tree/RelaxedTreeUtil.h>
#include <shogun/multiclass/tree/RelaxedTree.h>


using namespace shogun;

CRelaxedTree::CRelaxedTree()
	:m_feats(NULL), m_machine_for_confusion_matrix(NULL), m_num_classes(0)
{
}

CRelaxedTree::~CRelaxedTree()
{
	SG_UNREF(m_feats);
	SG_UNREF(m_machine_for_confusion_matrix);
}

CMulticlassLabels* CRelaxedTree::apply_multiclass(CFeatures* data)
{
	return NULL;
}

bool CRelaxedTree::train_machine(CFeatures* data)
{
	if (m_machine_for_confusion_matrix == NULL)
		SG_ERROR("Call set_machine_for_confusion_matrix before training");

	if (data)
	{
		CDenseFeatures<float64_t> *feats = dynamic_cast<CDenseFeatures<float64_t>*>(data);
		if (feats == NULL)
			SG_ERROR("Require non-NULL dense features of float64_t\n");
		set_features(feats);
	}

	CMulticlassLabels *lab = dynamic_cast<CMulticlassLabels *>(m_labels);

	RelaxedTreeUtil util;
	SGMatrix<float64_t> conf_mat = util.estimate_confusion_matrix(m_machine_for_confusion_matrix,
			m_feats, lab, m_num_classes);

	return false;
}

void CRelaxedTree::train_node(const SGMatrix<float64_t> &conf_mat, SGVector<int32_t> classes)
{

}

struct EntryComparator
{
	bool operator() (const CRelaxedTree::entry_t& e1, const CRelaxedTree::entry_t& e2)
	{
		return e1.second < e2.second;
	}
};
std::vector<CRelaxedTree::entry_t> CRelaxedTree::init_node(const SGMatrix<float64_t> &global_conf_mat, SGVector<int32_t> classes)
{
	// local confusion matrix
	SGMatrix<float64_t> conf_mat(classes.vlen, classes.vlen);
	for (index_t i=0; i < conf_mat.num_rows; ++i)
		for (index_t j=0; j < conf_mat.num_cols; ++j)
			conf_mat(i, j) = global_conf_mat(classes[i], classes[j]);

	// make conf matrix symmetry
	for (index_t i=0; i < conf_mat.num_rows; ++i)
		for (index_t j=0; j < conf_mat.num_cols; ++j)
			conf_mat(i,j) += conf_mat(j,i);

	int32_t num_entries = classes.vlen*(classes.vlen-1)/2;
	std::vector<CRelaxedTree::entry_t> entries;
	for (index_t i=0; i < classes.vlen; ++i)
		for (index_t j=i+1; j < classes.vlen; ++j)
			entries.push_back(std::make_pair(std::make_pair(i, j), conf_mat(i,j)));

	std::sort(entries.begin(), entries.end(), EntryComparator());

	const size_t max_n_samples = 30;
	int32_t n_samples = std::min(max_n_samples, entries.size());

	return std::vector<CRelaxedTree::entry_t>(entries.begin(), entries.begin() + n_samples);
}
