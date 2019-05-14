/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Philippe Tillet, Soeren Sonnenburg, Bjoern Esser, Sergey Lisitsyn
 */

#include <shogun/classifier/NearestCentroid.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/Features.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/features/iterators/DotIterator.h>



namespace shogun{

	CNearestCentroid::CNearestCentroid() : CDistanceMachine()
	{
		init();
	}

	CNearestCentroid::CNearestCentroid(CDistance* d, CLabels* trainlab) : CDistanceMachine()
	{
		init();
		ASSERT(d)
		ASSERT(trainlab)
		set_distance(d);
		set_labels(trainlab);
	}

	CNearestCentroid::~CNearestCentroid()
	{
	}

	void CNearestCentroid::init()
	{
		m_shrinking=0;
		m_is_trained=false;
	}


	bool CNearestCentroid::train_machine(CFeatures* data)
	{
		ASSERT(m_labels)
		ASSERT(m_labels->get_label_type() == LT_MULTICLASS)
		ASSERT(distance)
		if (data)
		{
			if (m_labels->get_num_labels() != data->get_num_vectors())
				SG_ERROR("Number of training vectors does not match number of labels\n")
			distance->init(data, data);
		}
		else
		{
			data = distance->get_lhs();
		}
		ASSERT(data->get_feature_class() == C_DENSE)

		auto* multiclass_labels = m_labels->as<CMulticlassLabels>();
		auto* dense_data = data->as<CDenseFeatures<float64_t>>();

		int32_t num_classes = multiclass_labels->get_num_classes();
		int32_t num_feats = dense_data->get_num_features();

		SGMatrix<float64_t> centroids(num_feats, num_classes);
		centroids.zero();

		SGVector<int64_t> num_per_class(num_classes);
		num_per_class.zero();

		auto iter_labels = multiclass_labels->get_int_labels().begin();
		for(const auto& current : DotIterator(dense_data))
		{
			const auto current_class = *(iter_labels++);
			auto target = centroids.get_column_vector(current_class);
			auto target_vec = SGVector<float64_t>(target, num_feats);
			current.add(1, target_vec);
			num_per_class[current_class]++;
		}

		for (int32_t i=0 ; i<num_classes ; i++)
		{
			auto* target = centroids.get_column_vector(i);
			int32_t total = num_per_class[i];
			float64_t scale = 0;
			if(total>1)
				scale = 1.0/((float64_t)(total-1));
			else
				scale = 1.0/(float64_t)total;

			SGVector<float64_t>::scale_vector(scale,target,num_feats);
		}

		auto centroids_feats = some<CDenseFeatures<float64_t>>(centroids);

		m_is_trained=true;
		distance->init(centroids_feats, distance->get_rhs());

		return true;
	}

}
