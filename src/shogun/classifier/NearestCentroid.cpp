/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Philippe Tillet
 */

#include "NearestCentroid.h"

namespace shogun{
	
  	CNearestCentroid::CNearestCentroid() : CDistanceMachine()
	{
		init();
	}

	CNearestCentroid::CNearestCentroid(CDistance* d, CLabels* trainlab) : CDistanceMachine()
	{
		init();
		ASSERT(d);
		ASSERT(trainlab);
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
		set_store_model_features(true);
		m_centroids = new CSimpleFeatures<float64_t>();
	}

	void CNearestCentroid::store_model_features()
	{

	}

	bool CNearestCentroid::train_machine(CFeatures* data)
	{
		ASSERT(m_labels);
		ASSERT(distance);
		
		if (data)
		{
			if (m_labels->get_num_labels() != data->get_num_vectors())
				SG_ERROR("Number of training vectors does not match number of labels\n");
			distance->init(data, data);
		}
		else
		{
			data = distance->get_lhs();
		}
		int32_t num_vectors = data->get_num_vectors();
		int32_t num_classes = m_labels->get_num_classes();
		int32_t num_feats = ((CSimpleFeatures<float64_t>*)data)->get_num_features();
		float64_t* centroids = SG_CALLOC(float64_t, num_feats*num_classes);
		for(int32_t i=0 ; i < num_feats*num_classes ; i++)
		{
			centroids[i]=0;
		}
		m_centroids->set_num_features(num_feats);
		m_centroids->set_num_vectors(num_classes);
		
		int64_t* num_per_class = new int64_t[num_classes];
		for(int32_t i=0 ; i<num_classes ; i++)
		{
			num_per_class[i]=0;
		}
		
		for(int32_t idx=0 ; idx<num_vectors ; idx++)
		{
			int32_t current_len;
			bool current_free;
			int32_t current_class = m_labels->get_label(idx);
			float64_t* target = centroids + num_feats*current_class;
			float64_t* current = ((CSimpleFeatures<float64_t>*)data)->get_feature_vector(idx,current_len,current_free);
			CMath::add(target,1.0,target,1.0,current,current_len);
			num_per_class[current_class]++;
			((CSimpleFeatures<float64_t>*)data)->free_feature_vector(current, current_len, current_free);
		}


		for(int32_t i=0 ; i<num_classes ; i++)
		{
			float64_t* target = centroids + num_feats*i;
			int32_t total = num_per_class[i];
			float64_t scale = 0;
			if(total>1)
				scale = 1.0/((float64_t)(total-1));
			else
				scale = 1.0/(float64_t)total;
				
			CMath::scale_vector(scale,target,num_feats);
		}
				
		m_centroids->free_feature_matrix();
		m_centroids->set_feature_matrix(centroids,num_feats,num_classes);

		
		distance->init(m_centroids,distance->get_rhs());
		
		m_is_trained=true;
		
		delete [] num_per_class;
		
		return true;
	}

}