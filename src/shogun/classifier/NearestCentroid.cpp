/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Philippe Tillet
 */

#include <shogun/classifier/NearestCentroid.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/Features.h>
#include <shogun/features/FeatureTypes.h>



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
		if(m_is_trained)
			distance->remove_lhs();
		else
			delete m_centroids;
	}

	void CNearestCentroid::init()
	{
		m_shrinking=0;
		m_is_trained=false;
		m_centroids = new CDenseFeatures<float64_t>();
	}


	bool CNearestCentroid::train_machine(CFeatures* data)
	{
		ASSERT(m_labels)
		ASSERT(m_labels->get_label_type() == LT_MULTICLASS)
		ASSERT(distance)
		ASSERT( data->get_feature_class() == C_DENSE)
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
		int32_t num_vectors = data->get_num_vectors();
		int32_t num_classes = ((CMulticlassLabels*) m_labels)->get_num_classes();
		int32_t num_feats = ((CDenseFeatures<float64_t>*) data)->get_num_features();
		SGMatrix<float64_t> centroids(num_feats,num_classes);
		centroids.zero();

		m_centroids->set_num_features(num_feats);
		m_centroids->set_num_vectors(num_classes);

		int64_t* num_per_class = new int64_t[num_classes];
		for (int32_t i=0 ; i<num_classes ; i++)
		{
			num_per_class[i]=0;
		}

		for (int32_t idx=0 ; idx<num_vectors ; idx++)
		{
			int32_t current_len;
			bool current_free;
			int32_t current_class = ((CMulticlassLabels*) m_labels)->get_label(idx);
			float64_t* target = centroids.matrix + num_feats*current_class;
			float64_t* current = ((CDenseFeatures<float64_t>*)data)->get_feature_vector(idx,current_len,current_free);
			SGVector<float64_t>::add(target,1.0,target,1.0,current,current_len);
			num_per_class[current_class]++;
			((CDenseFeatures<float64_t>*)data)->free_feature_vector(current, current_len, current_free);
		}


		for (int32_t i=0 ; i<num_classes ; i++)
		{
			float64_t* target = centroids.matrix + num_feats*i;
			int32_t total = num_per_class[i];
			float64_t scale = 0;
			if(total>1)
				scale = 1.0/((float64_t)(total-1));
			else
				scale = 1.0/(float64_t)total;

			SGVector<float64_t>::scale_vector(scale,target,num_feats);
		}

		m_centroids->free_feature_matrix();
		m_centroids->set_feature_matrix(centroids);


		m_is_trained=true;
		distance->init(m_centroids,distance->get_rhs());

		SG_FREE(num_per_class);

		return true;
	}

}
