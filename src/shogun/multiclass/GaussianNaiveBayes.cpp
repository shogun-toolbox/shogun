/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Viktor Gal, Giovanni De Toni, Soeren Sonnenburg, 
 *          Thoralf Klein, Björn Esser
 */

#include <shogun/base/progress.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/Signal.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/GaussianNaiveBayes.h>

#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

CGaussianNaiveBayes::CGaussianNaiveBayes() : CNativeMulticlassMachine(), m_features(NULL),
	m_min_label(0), m_num_classes(0), m_dim(0), m_means(), m_variances(),
	m_label_prob(), m_rates()
{
	init();
};

CGaussianNaiveBayes::CGaussianNaiveBayes(CFeatures* train_examples,
	CLabels* train_labels) : CNativeMulticlassMachine(), m_features(NULL),
	m_min_label(0), m_num_classes(0), m_dim(0), m_means(),
	m_variances(), m_label_prob(), m_rates()
{
	init();
	ASSERT(train_examples->get_num_vectors() == train_labels->get_num_labels())
	set_labels(train_labels);

	if (!train_examples->has_property(FP_DOT))
		SG_ERROR("Specified features are not of type CDotFeatures\n")

	set_features((CDotFeatures*)train_examples);
};

CGaussianNaiveBayes::~CGaussianNaiveBayes()
{
	SG_UNREF(m_features);
};

CFeatures* CGaussianNaiveBayes::get_features()
{
	SG_REF(m_features);
	return m_features;
}

void CGaussianNaiveBayes::set_features(CFeatures* features)
{
	if (!features->has_property(FP_DOT))
		SG_ERROR("Specified features are not of type CDotFeatures\n")

	SG_REF(features);
	SG_UNREF(m_features);
	m_features = (CDotFeatures*)features;
}

bool CGaussianNaiveBayes::train_machine(CFeatures* data)
{
	// init features with data if necessary and assure type is correct
	if (data)
	{
		if (!data->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features((CDotFeatures*) data);
	}

	// get int labels to train_labels and check length equality
	ASSERT(m_labels)
	ASSERT(m_labels->get_label_type() == LT_MULTICLASS)
	SGVector<int32_t> train_labels = ((CMulticlassLabels*) m_labels)->get_int_labels();
	ASSERT(m_features->get_num_vectors()==train_labels.vlen)

	// find minimal and maximal label
	auto min_label = CMath::min(train_labels.vector, train_labels.vlen);
	auto max_label = CMath::max(train_labels.vector, train_labels.vlen);
	int i,j;

	// subtract minimal label from all labels
	linalg::add_scalar(train_labels, -min_label);

	// get number of classes, minimal label and dimensionality
	m_num_classes = max_label-min_label+1;
	m_min_label = min_label;
	m_dim = m_features->get_dim_feature_space();

	// allocate memory for distributions' parameters and a priori probability
	m_means=SGMatrix<float64_t>(m_dim,m_num_classes);
	m_variances=SGMatrix<float64_t>(m_dim, m_num_classes);
	m_label_prob=SGVector<float64_t>(m_num_classes);

	// allocate memory for label rates
	m_rates=SGVector<float64_t>(m_num_classes);

	// make arrays filled by zeros before using
	m_means.zero();
	m_variances.zero();
	m_label_prob.zero();
	m_rates.zero();

	// number of iterations in all cycles
	int32_t max_progress = 2 * train_labels.vlen + 2 * m_num_classes;

	// Progress bar
	auto pb = progress(range(max_progress), *this->io);

	// get sum of features among labels
	for (i=0; i<train_labels.vlen; i++)
	{
		SGVector<float64_t> fea = m_features->get_computed_dot_feature_vector(i);
		for (j=0; j<m_dim; j++)
			m_means(j, train_labels.vector[i]) += fea.vector[j];

		m_label_prob.vector[train_labels.vector[i]]+=1.0;

		pb.print_progress();
	}

	// get means of features of labels
	for (i=0; i<m_num_classes; i++)
	{
		for (j=0; j<m_dim; j++)
			m_means(j, i) /= m_label_prob.vector[i];
		pb.print_progress();
	}

	// compute squared residuals with means available
	for (i=0; i<train_labels.vlen; i++)
	{
		SGVector<float64_t> fea = m_features->get_computed_dot_feature_vector(i);
		for (j=0; j<m_dim; j++)
		{
			m_variances(j, train_labels.vector[i]) +=
				CMath::sq(fea[j]-m_means(j, train_labels.vector[i]));
		}
		pb.print_progress();
	}

	// get variance of features of labels
	for (i=0; i<m_num_classes; i++)
	{
		for (j=0; j<m_dim; j++)
			m_variances(j, i) /= m_label_prob.vector[i] > 1 ? m_label_prob.vector[i]-1 : 1;

		// get a priori probabilities of labels
		m_label_prob.vector[i]/= m_num_classes;

		pb.print_progress();
	}
	pb.complete();

	return true;
}

CMulticlassLabels* CGaussianNaiveBayes::apply_multiclass(CFeatures* data)
{
	if (data)
		set_features(data);

	ASSERT(m_features)

	// init number of vectors
	int32_t num_vectors = m_features->get_num_vectors();

	// init result labels
	CMulticlassLabels* result = new CMulticlassLabels(num_vectors);

	// classify each example of data
	for (auto i : progress(range(num_vectors), *this->io))
	{
		result->set_label(i,apply_one(i));
	}

	return result;
};

float64_t CGaussianNaiveBayes::apply_one(int32_t idx)
{
	// get [idx] feature vector
	SGVector<float64_t> feature_vector = m_features->get_computed_dot_feature_vector(idx);

	// init loop variables
	int i,k;

	// rate all labels
	for (i=0; i<m_num_classes; i++)
	{
		// set rate to 0.0 if a priori probability is 0.0 and continue
		if (m_label_prob.vector[i]==0.0)
		{
			m_rates.vector[i] = 0.0;
			continue;
		}
		else
			m_rates.vector[i] = std::log(m_label_prob.vector[i]);

		// product all conditional gaussian probabilities
		for (k=0; k<m_dim; k++)
			if (m_variances(k,i)!=0.0)
				m_rates.vector[i] +=
				    std::log(0.39894228 / CMath::sqrt(m_variances(k, i))) -
				    0.5 * CMath::sq(feature_vector.vector[k] - m_means(k, i)) /
				        (m_variances(k, i));
	}

	// find label with maximum rate
	int32_t max_label_idx = 0;

	for (i=0; i<m_num_classes; i++)
	{
		if (m_rates.vector[i]>m_rates.vector[max_label_idx])
			max_label_idx = i;
	}

	return max_label_idx+m_min_label;
};

void CGaussianNaiveBayes::init()
{
	SG_ADD(&m_min_label, "m_min_label", "minimal label", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_classes, "m_num_classes",
		"number of different classes (labels)", MS_NOT_AVAILABLE);
	SG_ADD(&m_dim, "m_dim",
		"dimensionality of feature space", MS_NOT_AVAILABLE);
	SG_ADD(&m_means, "m_means",
		"means for normal distributions of features", MS_NOT_AVAILABLE);
	SG_ADD(&m_variances, "m_variances",
		"variances for normal distributions of features", MS_NOT_AVAILABLE);
	SG_ADD(&m_label_prob, "m_label_prob",
		"a priori probabilities of labels", MS_NOT_AVAILABLE);
	SG_ADD(&m_rates, "m_rates", "label rates", MS_NOT_AVAILABLE);
}