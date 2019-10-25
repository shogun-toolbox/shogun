/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/evaluation/MultilabelAccuracy.h>
#include <shogun/labels/MultilabelLabels.h>

using namespace shogun;

MultilabelAccuracy::MultilabelAccuracy()
    : Evaluation()
{
}

MultilabelAccuracy::~MultilabelAccuracy()
{
}

float64_t MultilabelAccuracy::evaluate(std::shared_ptr<Labels> predicted,
        std::shared_ptr<Labels> ground_truth)
{
    require(predicted->get_label_type() == LT_SPARSE_MULTILABEL,
            "predicted label should be of multilabels type");
    require(ground_truth->get_label_type() == LT_SPARSE_MULTILABEL,
            "actual label should be of multilabels type");
    require(ground_truth->get_label_type() == predicted->get_label_type(),
            "predicted labels and actual labels should be of same type");

    auto m_predicted = std::static_pointer_cast<MultilabelLabels>(predicted);
    auto m_ground_truth = std::static_pointer_cast<MultilabelLabels>(ground_truth);

    require(m_predicted->get_num_labels() == m_ground_truth->get_num_labels(),
            "predicted labels and actual labels should have same number of labels");
    require(m_predicted->get_num_classes() == m_ground_truth->get_num_classes(),
            "predicted labels and actual labels should have same number of classes");

    int32_t num_labels = predicted->get_num_labels();
    float64_t accuracy = 0.0;

    for (index_t k=0; k<num_labels; k++)
    {
        SGVector<int32_t> slabel_true = m_ground_truth->get_label(k);
        SGVector<int32_t> slabel_pred = m_predicted->get_label(k);

        int32_t true_pos = 0;
        index_t i = 0, j = 0;

        while (i<slabel_true.vlen && j<slabel_pred.vlen)
        {
            /** true positive */
            if (slabel_true[i] == slabel_pred[j])
            {
                true_pos ++;
                i++;
                j++;
            }
            /** false positive */
            else if (slabel_true[i] < slabel_pred[j])
            {
                i++;
            }
            /** false negative */
            else
            {
                j++;
            }
        }

        accuracy += ((float)true_pos /
                     (float)(slabel_true.vlen + slabel_pred.vlen - true_pos));
    }

    return accuracy/num_labels;
}

