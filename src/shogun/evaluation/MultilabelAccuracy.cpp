/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/evaluation/MultilabelAccuracy.h>
#include <shogun/labels/MultilabelLabels.h>

using namespace shogun;

CMultilabelAccuracy::CMultilabelAccuracy()
    : CEvaluation()
{
}

CMultilabelAccuracy::~CMultilabelAccuracy()
{
}

float64_t CMultilabelAccuracy::evaluate(CLabels * predicted,
        CLabels * ground_truth)
{
	float64_t accuracy, score;
	SGVector<int32_t> tp, fp, fn;

	evaluate(
			predicted,
			ground_truth,
			accuracy,
			score,
			tp,
			fp,
			fn);
	return accuracy;
}

float64_t CMultilabelAccuracy::get_f1_score(CLabels * predicted,
		CLabels * ground_truth)
{
	float64_t accuracy, score;
	SGVector<int32_t> tp, fp, fn;

	evaluate(
			predicted,
			ground_truth,
			accuracy,
			score,
			tp,
			fp,
			fn);
	return score;
}

SGVector<int32_t> CMultilabelAccuracy::get_true_pos(CLabels * predicted,
		CLabels * ground_truth)
{
	float64_t accuracy, score;
	SGVector<int32_t> tp, fp, fn;

	evaluate(
			predicted,
			ground_truth,
			accuracy,
			score,
			tp,
			fp,
			fn);
	return tp;
}

SGVector<int32_t> CMultilabelAccuracy::get_false_pos(CLabels * predicted,
		CLabels * ground_truth)
{
	float64_t accuracy, score;
	SGVector<int32_t> tp, fp, fn;

	evaluate(
			predicted,
			ground_truth,
			accuracy,
			score,
			tp,
			fp,
			fn);
	return fp;
}

SGVector<int32_t> CMultilabelAccuracy::get_false_neg(CLabels * predicted,
		CLabels * ground_truth)
{
	float64_t accuracy, score;
	SGVector<int32_t> tp, fp, fn;

	evaluate(
			predicted,
			ground_truth,
			accuracy,
			score,
			tp,
			fp,
			fn);
	return fn;
}

void CMultilabelAccuracy::evaluate(
		CLabels* predicted,
        CLabels* ground_truth,
		float64_t & accuracy,
		float64_t & score,
		SGVector<int32_t> & tp,
		SGVector<int32_t> & fp,
		SGVector<int32_t> & fn)
{
    REQUIRE(predicted->get_label_type() == LT_SPARSE_MULTILABEL,
            "predicted label should be of multilabels type\n");
    REQUIRE(ground_truth->get_label_type() == LT_SPARSE_MULTILABEL,
            "actual label should be of multilabels type\n");
    REQUIRE(ground_truth->get_label_type() == predicted->get_label_type(),
            "predicted labels and actual labels should be of same type\n");

    CMultilabelLabels* m_predicted = (CMultilabelLabels*) predicted;
    CMultilabelLabels* m_ground_truth = (CMultilabelLabels*) ground_truth;

    REQUIRE(m_predicted->get_num_labels() == m_ground_truth->get_num_labels(),
            "predicted labels and actual labels should have same number of labels\n");
    REQUIRE(m_predicted->get_num_classes() == m_ground_truth->get_num_classes(),
            "predicted labels and actual labels should have same number of classes\n");

    int32_t num_labels = predicted->get_num_labels();

    accuracy = 0.0;
	score = 0.0;
	tp = SGVector<int32_t>(num_labels);
	fp = SGVector<int32_t>(num_labels);
	fn = SGVector<int32_t>(num_labels);

	float64_t precision = 0.0;
	float64_t recall = 0.0;

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
		precision += ((float)true_pos / (float)slabel_pred.vlen);
		recall += ((float)true_pos / (float)slabel_true.vlen);
		tp[k] = true_pos;
		fp[k] = (slabel_pred.vlen - true_pos);
		fn[k] = (slabel_true.vlen - true_pos);
    }

    accuracy /= num_labels;
	precision /= num_labels;
	recall /= num_labels;
	score = (2 * (precision * recall)) / (precision + recall);
}
