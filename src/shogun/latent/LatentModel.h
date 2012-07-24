/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __LATENTMODEL_H__
#define __LATENTMODEL_H__

#include <shogun/labels/LatentLabels.h>
#include <shogun/features/LatentFeatures.h>

namespace shogun
{
	/** @brief TODO
	 */
	class CLatentModel: public CSGObject
	{
		public:
			/** default ctor */
			CLatentModel();

			/** ctor
			 *
			 * @param feats Latent features
			 * @param labels Latent labels
			 */
			CLatentModel(CLatentFeatures* feats, CLatentLabels* labels);

			/** destructor */
			virtual ~CLatentModel();

			/**
			 * get the number of examples
			 *
			 * @return number of examples/vectors in latent features
			 */
			virtual int32_t get_num_vectors() const;

			/**
			 * get the dimension of PSI
			 *
			 * @return dimension of features, i.e. psi vector
			 */
			virtual int32_t get_dim() const=0;

			/** set labels
			 *
			 * @param labs labels
			 */
			void set_labels(CLatentLabels* labs);

			/** get labels
			 *
			 * @return latent labels
			 */
			CLatentLabels* get_labels() const;

			/** set features
			 *
			 * @param feats features
			 */
			void set_features(CLatentFeatures* feats);

			/**
			 * Calculate the PSI vector for a given sample
			 *
			 * @param idx index of the sample
			 *
			 * @return PSI vector
			 */
			virtual SGVector<float64_t> get_psi_feature_vector(index_t idx)=0;

			/** infer latent variable
			 *
			 * @param w weight vector
			 * @param idx index of feature vector
			 * @return latent variable data
			 */
			virtual CLatentData* infer_latent_variable(const SGVector<float64_t>& w, index_t idx)=0;

			/** argmax 
			 *
			 * @param w weight vector
			 */
			virtual void argmax_h(const SGVector<float64_t>& w);

			/** get name */
			virtual const char* get_name() const { return "LatentModel"; }

		protected:
			/** features */
			CLatentFeatures* m_features;
			/** labels */
			CLatentLabels* m_labels;

		private:
			void register_parameters();
	};
}

#endif /* __LATENTMODEL_H__ */

