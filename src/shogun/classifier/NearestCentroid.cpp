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
#include <shogun/mathematics/linalg/LinalgNamespace.h>


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
		ASSERT(distance)
		if (data)
		{
			if (m_labels->get_num_labels() != data->get_num_vectors())
				error("Number of training vectors does not match number of labels\n");
			distance->init(data, data);
		}
		else
		{
			data = distance->get_lhs();
		}

		auto* multiclass_labels = m_labels->as<CMulticlassLabels>();
		auto* dense_data = data->as<CDenseFeatures<float64_t>>();

		int32_t num_classes = multiclass_labels->get_num_classes();
		int32_t num_feats = dense_data->get_num_features();

		SGMatrix<float64_t> centroids(num_feats, num_classes);
		SGVector<int64_t> num_per_class(num_classes);

		auto iter_labels = multiclass_labels->get_labels().begin();
		for (const auto& current : DotIterator(dense_data))
		{
			const auto current_class = static_cast<int32_t>(*(iter_labels++));
			auto target = centroids.get_column(current_class);
			current.add(1, target);
			num_per_class[current_class]++;
		}

		SGVector<float64_t> scale(num_classes);
		for (int32_t i=0 ; i<num_classes ; i++)
		{
			int32_t total = num_per_class[i];
			if(total>1)
				scale[i] = 1.0/((float64_t)(total-1));
			else
				scale[i] = 1.0/(float64_t)total;
		}
		linalg::scale(centroids, centroids, scale);

		auto centroids_feats = some<CDenseFeatures<float64_t>>(centroids);

		m_is_trained=true;
		distance->init(centroids_feats, distance->get_rhs());

		return true;
	}

}
