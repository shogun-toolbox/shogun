/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <algorithm>

#include <mathematics/Math.h>
#include <labels/BinaryLabels.h>
#include <labels/MulticlassLabels.h>
#include <multiclass/ecoc/ECOCDiscriminantEncoder.h>

using namespace std;
using namespace shogun;

CECOCDiscriminantEncoder::CECOCDiscriminantEncoder()
{
    init();
}

CECOCDiscriminantEncoder::~CECOCDiscriminantEncoder()
{
    SG_UNREF(m_features);
    SG_UNREF(m_labels);
}

void CECOCDiscriminantEncoder::init()
{
    // default parameters
    m_iterations = 25;
    m_num_trees = 1;

    // init values
    m_features = NULL;
    m_labels = NULL;

    // parameters

    SG_ADD(&m_iterations, "iterations", "number of iterations in SFFS", MS_NOT_AVAILABLE);
}

void CECOCDiscriminantEncoder::set_features(CDenseFeatures<float64_t> *features)
{
    SG_REF(features);
    SG_UNREF(m_features);
    m_features = features;
}

void CECOCDiscriminantEncoder::set_labels(CLabels *labels)
{
    SG_REF(labels);
    SG_UNREF(m_labels);
    m_labels = labels;
}

SGMatrix<int32_t> CECOCDiscriminantEncoder::create_codebook(int32_t num_classes)
{
    if (!m_features || !m_labels)
        SG_ERROR("Need features and labels to learn the codebook")

    m_feats = m_features->get_feature_matrix();
    m_codebook = SGMatrix<int32_t>(m_num_trees * (num_classes-1), num_classes);
    m_codebook.zero();
    m_code_idx = 0;

    for (int32_t itree = 0; itree < m_num_trees; ++itree)
    {
        vector<int32_t> classes(num_classes);
        for (int32_t i=0; i < num_classes; ++i)
            classes[i] = i;

        binary_partition(classes);
    }

    m_feats = SGMatrix<float64_t>(); // release memory
    return m_codebook;
}

void CECOCDiscriminantEncoder::binary_partition(const vector<int32_t>& classes)
{
    if (classes.size() > 2)
    {
        int32_t isplit = classes.size()/2;
        vector<int32_t> part1(classes.begin(), classes.begin()+isplit);
        vector<int32_t> part2(classes.begin()+isplit, classes.end());
        run_sffs(part1, part2);
        for (size_t i=0; i < part1.size(); ++i)
            m_codebook(m_code_idx, part1[i]) = +1;
        for (size_t i=0; i < part2.size(); ++i)
            m_codebook(m_code_idx, part2[i]) = -1;
        m_code_idx++;

        if (part1.size() > 1)
            binary_partition(part1);
        if (part2.size() > 1)
            binary_partition(part2);
    }
    else // only two classes
    {
        m_codebook(m_code_idx, classes[0]) = +1;
        m_codebook(m_code_idx, classes[1]) = -1;
        m_code_idx++;
    }
}

void CECOCDiscriminantEncoder::run_sffs(vector<int32_t>& part1, vector<int32_t>& part2)
{
    set<int32_t> idata1;
    set<int32_t> idata2;

    for (int32_t i=0; i < m_labels->get_num_labels(); ++i)
    {
        if (find(part1.begin(), part1.end(), ((CMulticlassLabels*) m_labels)->get_int_label(i)) != part1.end())
            idata1.insert(i);
        else if (find(part2.begin(), part2.end(), ((CMulticlassLabels*) m_labels)->get_int_label(i)) != part2.end())
            idata2.insert(i);
    }

    float64_t MI = compute_MI(idata1, idata2);
    for (int32_t i=0; i < m_iterations; ++i)
    {
        if (i % 2 == 0)
            MI = sffs_iteration(MI, part1, idata1, part2, idata2);
        else
            MI = sffs_iteration(MI, part2, idata2, part1, idata1);
    }
}

float64_t CECOCDiscriminantEncoder::sffs_iteration(float64_t MI, vector<int32_t>& part1, set<int32_t>& idata1,
        vector<int32_t>& part2, set<int32_t>& idata2)
{
    if (part1.size() <= 1)
        return MI;

    int32_t iclas = CMath::random(0, int32_t(part1.size()-1));
    int32_t clas = part1[iclas];

    // move clas from part1 to part2
    for (int32_t i=0; i < m_labels->get_num_labels(); ++i)
    {
        if (((CMulticlassLabels*) m_labels)->get_int_label(i) == clas)
        {
            idata1.erase(i);
            idata2.insert(i);
        }
    }

    float64_t new_MI = compute_MI(idata1, idata2);
    if (new_MI < MI)
    {
        part2.push_back(clas);
        part1.erase(part1.begin() + iclas);
        return new_MI;
    }
    else
    {
        // revert changes
        for (int32_t i=0; i < m_labels->get_num_labels(); ++i)
        {
            if (((CMulticlassLabels*) m_labels)->get_int_label(i) == clas)
            {
                idata2.erase(i);
                idata1.insert(i);
            }
        }
        return MI;
    }

}

float64_t CECOCDiscriminantEncoder::compute_MI(const set<int32_t>& idata1, const set<int32_t>& idata2)
{
    float64_t MI = 0;

    int32_t hist1[10];
    int32_t hist2[10];

    for (int32_t i=0; i < m_feats.num_rows; ++i)
    {
        float64_t max_val = m_feats(i, 0);
        float64_t min_val = m_feats(i, 0);
        for (int32_t j=1; j < m_feats.num_cols; ++j)
        {
            max_val = max(max_val, m_feats(i, j));
            min_val = min(min_val, m_feats(i, j));
        }

        if (max_val - min_val < 1e-10)
            max_val = min_val + 1; // avoid divide by zero error

        compute_hist(i, max_val, min_val, idata1, hist1);
        compute_hist(i, max_val, min_val, idata2, hist2);

        float64_t MI_i = 0;
        for (int j=0; j < 10; ++j)
            MI_i += (hist1[j]-hist2[j])*(hist1[j]-hist2[j]);
        MI += CMath::sqrt(MI_i);
    }

    return MI;
}

void CECOCDiscriminantEncoder::compute_hist(int32_t i, float64_t max_val, float64_t min_val,
        const set<int32_t>& idata, int32_t *hist)
{
    // hist of 0:0.1:1
    fill(hist, hist+10, 0);

    for (set<int32_t>::const_iterator it = idata.begin(); it != idata.end(); ++it)
    {
        float64_t val = (m_feats(i, *it) - min_val) / (max_val - min_val);
        int32_t pos = min(9, static_cast<int32_t>(val*10));
        hist[pos]++;
    }
}
