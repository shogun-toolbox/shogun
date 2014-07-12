/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/labels/Labels.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/preprocessor/FeatureSelection.h>

namespace shogun
{

template <class ST>
CFeatureSelection<ST>::CFeatureSelection() : CPreprocessor()
{
	init();
}

template <class ST>
void CFeatureSelection<ST>::init()
{
	SG_ADD(&m_target_dim, "target_dim", "target dimension",
			MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*)&m_algorithm, "algorithm",
			"the feature selectiona algorithm", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*)&m_policy, "policy", "feature removal policy",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_num_remove, "num_remove", "number or percentage of features to "
			"be removed", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_labels, "labels",
			"the class labels for the features", MS_NOT_AVAILABLE);

	m_target_dim=0;
	m_algorithm=BACKWARD_ELIMINATION;
	m_policy=N_LARGEST;
	m_num_remove=1;
	m_labels=NULL;
}

template <class ST>
CFeatureSelection<ST>::~CFeatureSelection()
{
	SG_UNREF(m_labels);
}


template <class ST>
void CFeatureSelection<ST>::cleanup()
{
}

template <class ST>
CFeatures* CFeatureSelection<ST>::apply_backward_elimination(CFeatures* features)
{
	SG_DEBUG("Entering!\n");

	// precompute whenever appropriate for performing the rest of the tasks
	precompute();

	// NULL check for features is handled in get_num_features
	index_t num_features=get_num_features(features);
	SG_DEBUG("Initial number of features %d!\n", num_features);

	// the main loop
	while (num_features>m_target_dim)
	{
		// tune the measurement parameters whenever necessary based on current
		// features
		adapt_params(features);

		// compute the measures for each of the current dimensions
		SGVector<float64_t> measures(num_features);
		for (index_t i=0; i<num_features; ++i)
			measures[i]=compute_measures(features, i);

		if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
			measures.display_vector("measures");

		// rank the measures
		SGVector<index_t> argsorted=measures.argsort();

		if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
			argsorted.display_vector("argsorted");

		// make sure that we don't end up with lesser feats than target dim
		index_t to_remove;
		if (m_policy==N_SMALLEST || m_policy==N_LARGEST)
			to_remove=m_num_remove;
		else
			to_remove=num_features*m_num_remove*0.01;

		index_t can_remove=num_features-m_target_dim;

		// if policy is to remove N feats corresponding to smallest/largest
		// measures, we just replace N with can_remove. if policy is to remove
		// N% feats, then we change the policy temporarily and remove a fixed
		// can_remove number of feats instead
		index_t orig_remove=m_num_remove;
		EFeatureRemovalPolicy orig_policy=m_policy;

		if (to_remove>can_remove)
		{
			m_num_remove=can_remove;
			SG_DEBUG("Can only remove %d features in this iteration!\n",
					can_remove);

			if (m_policy==PERCENTILE_SMALLEST)
				m_policy=N_SMALLEST;
			else if (m_policy==PERCENTILE_LARGEST)
				m_policy=N_LARGEST;
		}

		// remove appropriate number of features based on the measures and the
		// removal policy
		features=remove_feats(features, argsorted);

		// restore original removal policy and numbers if necessary for the
		// sake of consistency
		if (to_remove>can_remove)
		{
			m_policy=orig_policy;
			m_num_remove=orig_remove;
		}

		// update the number of features
		num_features=get_num_features(features);
		SG_DEBUG("Current number of features %d!\n", num_features);
	}

	SG_DEBUG("Leaving!\n");
	return features;
}

template <class ST>
CFeatures* CFeatureSelection<ST>::apply(CFeatures* features)
{
	SG_DEBUG("Entering!\n");

	// sanity checks
	REQUIRE(features, "Features cannot be NULL!\n");
	REQUIRE(features->get_num_vectors()>0,
			"Number of feature vectors has to be positive!\n");
	REQUIRE(m_target_dim>0, "Target dimension (%d) has to be positive! Set "
			"a higher number via set_target_dim().\n", m_target_dim);

	index_t num_features=get_num_features(features);
	REQUIRE(num_features>0, "Invalid number of features (%d)! Most likely "
			"feature selection cannot be performed for %s!\n",
			num_features, features->get_name());
	REQUIRE(num_features>m_target_dim,
			"Number of original features (dimensions of the feature vectors) "
			"(%d) has to be greater that the target dimension (%d)!\n",
			num_features, m_target_dim);

	// this method makes a deep copy of the feature object and performs
	// feature selection on it. This is already SG_REF'ed because of the
	// implementation of clone()
	CFeatures* feats_copy=(CFeatures*)features->clone();

	switch (m_algorithm)
	{
		case BACKWARD_ELIMINATION:
			return apply_backward_elimination(feats_copy);
		default:
			SG_ERROR("Specified algorithm not yet supported!\n");
			return features;
	}

	SG_DEBUG("Leaving!\n");
}

template <class ST>
void CFeatureSelection<ST>::precompute()
{
}

template <class ST>
void CFeatureSelection<ST>::adapt_params(CFeatures* features)
{
}

template <class ST>
index_t CFeatureSelection<ST>::get_num_features(CFeatures* features) const
{
	REQUIRE(features, "Features not initialized!\n");

	EFeatureClass f_class=features->get_feature_class();

	switch (f_class)
	{
		case C_DENSE:
		{
			CDenseFeatures<ST>* d_feats=dynamic_cast<CDenseFeatures<ST>*>(features);
			REQUIRE(d_feats, "Type mismatch for dense features!\n");
			return d_feats->get_num_features();
		}
		case C_SPARSE:
		{
			CSparseFeatures<ST>* s_feats=dynamic_cast<CSparseFeatures<ST>*>(features);
			REQUIRE(s_feats, "Type mismatch for sparse features!\n");
			return s_feats->get_num_features();
		}
		default:
			SG_ERROR("Number of features not available for %s!\n",
					features->get_name());
			break;
	}

	return 0;
}

template <class ST>
void CFeatureSelection<ST>::set_target_dim(index_t target_dim)
{
	m_target_dim=target_dim;
}

template <class ST>
index_t CFeatureSelection<ST>::get_target_dim() const
{
	return m_target_dim;
}

template <class ST>
EFeatureSelectionAlgorithm CFeatureSelection<ST>::get_algorithm() const
{
	return m_algorithm;
}

template <class ST>
EFeatureRemovalPolicy CFeatureSelection<ST>::get_policy() const
{
	return m_policy;
}

template <class ST>
void CFeatureSelection<ST>::set_num_remove(index_t num_remove)
{
	m_num_remove=num_remove;
}

template <class ST>
index_t CFeatureSelection<ST>::get_num_remove() const
{
	return m_num_remove;
}

template <class ST>
void CFeatureSelection<ST>::set_labels(CLabels* labels)
{
	SG_REF(labels);
	SG_UNREF(m_labels);
	m_labels=labels;
}

template <class ST>
CLabels* CFeatureSelection<ST>::get_labels() const
{
	SG_REF(m_labels);
	return m_labels;
}

template <class ST>
EFeatureClass CFeatureSelection<ST>::get_feature_class()
{
	return C_ANY;
}

template <class ST>
EPreprocessorType CFeatureSelection<ST>::get_type() const
{
	return P_UNKNOWN;
}

template<>
EFeatureType CFeatureSelection<floatmax_t>::get_feature_type()
{
	return F_LONGREAL;
}

template<>
EFeatureType CFeatureSelection<float64_t>::get_feature_type()
{
	return F_DREAL;
}

template<>
EFeatureType CFeatureSelection<float32_t>::get_feature_type()
{
	return F_SHORTREAL;
}

template<>
EFeatureType CFeatureSelection<int16_t>::get_feature_type()
{
	return F_SHORT;
}

template<>
EFeatureType CFeatureSelection<uint16_t>::get_feature_type()
{
	return F_WORD;
}

template<>
EFeatureType CFeatureSelection<char>::get_feature_type()
{
	return F_CHAR;
}

template<>
EFeatureType CFeatureSelection<int8_t>::get_feature_type()
{
	return F_CHAR;
}

template<>
EFeatureType CFeatureSelection<uint8_t>::get_feature_type()
{
	return F_BYTE;
}

template<>
EFeatureType CFeatureSelection<int32_t>::get_feature_type()
{
	return F_INT;
}

template<>
EFeatureType CFeatureSelection<uint32_t>::get_feature_type()
{
	return F_UINT;
}

template<>
EFeatureType CFeatureSelection<int64_t>::get_feature_type()
{
	return F_LONG;
}

template<>
EFeatureType CFeatureSelection<uint64_t>::get_feature_type()
{
	return F_ULONG;
}

template<>
EFeatureType CFeatureSelection<bool>::get_feature_type()
{
	return F_BOOL;
}

template class CFeatureSelection<bool>;
template class CFeatureSelection<char>;
template class CFeatureSelection<int8_t>;
template class CFeatureSelection<uint8_t>;
template class CFeatureSelection<int16_t>;
template class CFeatureSelection<uint16_t>;
template class CFeatureSelection<int32_t>;
template class CFeatureSelection<uint32_t>;
template class CFeatureSelection<int64_t>;
template class CFeatureSelection<uint64_t>;
template class CFeatureSelection<float32_t>;
template class CFeatureSelection<float64_t>;
template class CFeatureSelection<floatmax_t>;

}
