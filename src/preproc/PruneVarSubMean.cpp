#ifdef PRUNE_VAR_SUB_MEAN

void CHMM::subtract_mean_from_top_feature_cache(int num_features, int totobs)
{
	if (feature_cache_obs)
	{
		for (int j=0; j<num_features; j++)
		{
			double mean=0;
			for (int i=0; i<totobs; i++)
				mean+=feature_cache_obs[i*num_features+j];
			for (int i=0; i<totobs; i++)
				feature_cache_obs[i*num_features+j]-=mean;
		}
	}
}

#endif
