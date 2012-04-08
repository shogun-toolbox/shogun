#include "NearestCentroid.h"

namespace shogun{
	
	CLabels* CNearestCentroid::apply()
	{
		ASSERT(distance);
		ASSERT(distance->get_num_vec_rhs());
		int32_t n_centroids=m_centroids->get_num_vectors();
		ASSERT(m_centroids->get_num_vectors()>0);
		
		int32_t num_lab=distance->get_num_vec_rhs();
		CLabels* output=new CLabels(num_lab);
		for(int32_t idx = 0 ; idx<num_lab ; ++idx){
			output->set_label(idx,CDistanceMachine::apply(idx));
		}
		return output;
	}

	CLabels* CNearestCentroid::apply(CFeatures* data)
	{
		distance->replace_rhs(data);
		return apply();
	}

	float64_t CNearestCentroid::apply(int32_t vec_idx)
	{
		return CDistanceMachine::apply(vec_idx);
	}


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

	void CNearestCentroid::init()
	{
		m_shrinking=0;
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
		m_centroids->free_feature_matrix();
		float64_t* centroids = SG_MALLOC(float64_t, num_feats*num_classes);
		m_centroids->set_num_features(num_feats);
		m_centroids->set_num_vectors(num_classes);
		m_centroids->set_feature_matrix(centroids,num_feats,num_classes);
		
		int64_t* num_per_class = new int64_t[num_classes];
		for(int32_t i=0 ; i<num_classes ; ++i){
			num_per_class[i]=0;
		}
		
		for(int32_t idx=0 ; idx<num_vectors ; ++idx){
			int32_t target_len, current_len;
			bool target_free, current_free;
			int32_t current_class = m_labels->get_label(idx);
			float64_t* target = m_centroids->get_feature_vector(current_class,target_len,target_free);
			float64_t* current = ((CSimpleFeatures<float64_t>*)data)->get_feature_vector(idx,current_len,current_free);
			CMath::add(target,1.0,target,1.0,current,current_len);
			++num_per_class[current_class];
		}

		
		for(int32_t i=0 ; i<num_classes ; ++i){
			int32_t target_len;
			bool target_free;
			float64_t* target = m_centroids->get_feature_vector(i,target_len,target_free);
			CMath::scale_vector(1.0/(float64_t)num_per_class[i],target,target_len);
		}
		
		distance->init(m_centroids,distance->get_rhs());

		return true;
	}

}