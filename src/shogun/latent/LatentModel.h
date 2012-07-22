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
	class CLatentModel: public CSGObject
	{
		public:
			CLatentModel();

			CLatentModel(CLatentFeatures* feats, CLatentLabels* labels);

			virtual ~CLatentModel();

			virtual int32_t get_num_vectors() const;

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

			virtual SGVector<float64_t> get_psi_feature_vector(index_t idx)=0;

			virtual CLatentData* infer_latent_variable(const SGVector<float64_t>& w, index_t idx)=0;

			virtual void argmax_h(const SGVector<float64_t>& w);

			virtual const char* get_name() const { return "LatentModel"; }

		protected:
			CLatentFeatures* m_features;
			CLatentLabels* m_labels;

		private:
			void register_parameters();
	};
}

#endif /* __LATENTMODEL_H__ */

