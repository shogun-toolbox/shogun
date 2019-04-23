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

	NearestCentroid::NearestCentroid() : DistanceMachine()
	{
		init();
	}

	NearestCentroid::NearestCentroid(std::shared_ptr<Distance> d, std::shared_ptr<Labels> trainlab) : DistanceMachine()
	{
		init();
		ASSERT(d)
		ASSERT(trainlab)
		set_distance(d);
		set_labels(trainlab);
	}

	NearestCentroid::~NearestCentroid()
	{
	}

	void NearestCentroid::init()
	{
		m_shrinking=0;
		m_is_trained=false;
	}


	bool NearestCentroid::train_machine(std::shared_ptr<Features> data)
	{
		ASSERT(m_labels)
		ASSERT(distance)
		if (data)
		{
			if (m_labels->get_num_labels() != data->get_num_vectors())
				error("Number of training vectors does not match number of labels");
			distance->init(data, data);
		}
		else
		{
			data = distance->get_lhs();
		}

		auto multiclass_labels = m_labels->as<MulticlassLabels>();
		auto dense_data = data->as<DenseFeatures<float64_t>>();

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

		auto centroids_feats = std::make_shared<DenseFeatures<float64_t>>(centroids);

		m_is_trained=true;
		distance->init(centroids_feats, distance->get_rhs());

		return true;
	}

}
