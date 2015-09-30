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

#include <shogun/lib/config.h>

#include <shogun/labels/LatentLabels.h>
#include <shogun/features/LatentFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
	/** @brief Abstract class CLatentModel
	 * It represents the application specific model and contains most of the
	 * application dependent logic to solve latent variable based problems.
	 *
	 * The idea is that the user have to define and implement her own model, which
	 * is derived from CLatentModel and implement all the pure virtual functions
	 * which depends on the given problem she wants to solve, like the combined
	 * feature representation: \f$\Psi(\bold{x_i},\bold{h_i})\f$ and the inference
	 * of the latent variable \f$argmax_{h} \langle \bold{w},\Psi(\bold{x},\bold{h}) \rangle\f$
	 */
	class CLatentModel: public CSGObject
	{
		public:
			/** default ctor */
			CLatentModel();

			/** constructor
			 *
			 * @param feats Latent features
			 * @param labels Latent labels
			 * @param do_caching whether caching of PSI vectors is enabled or not. Enabled by default.
			 */
			CLatentModel(CLatentFeatures* feats, CLatentLabels* labels, bool do_caching = true);

			/** destructor */
			virtual ~CLatentModel();

			/** get the number of examples
			 *
			 * @return number of examples/vectors in latent features
			 */
			virtual int32_t get_num_vectors() const;

			/** get the dimension of the combined features, i.e \f$\Psi(\ldots)\f$
			 *
			 * @return dimension of features, i.e. psi vector
			 */
			virtual int32_t get_dim() const=0;

			/** set latent labels
			 *
			 * @param labs latent labels
			 */
			void set_labels(CLatentLabels* labs);

			/** get latent labels
			 *
			 * @return latent labels
			 */
			CLatentLabels* get_labels() const;

			/** set latent features
			 *
			 * @param feats the latent features of the problem
			 */
			void set_features(CLatentFeatures* feats);

			/** get latent features
			 *
			 * @return latent features
			 */
			CLatentFeatures* get_features() const;

			/** Calculate the PSI vectors for all features
			 *
			 * @return PSI vectors
			 */
			virtual CDotFeatures* get_psi_feature_vectors()=0;

			/** User defined \f$h^{*} = argmax_{h} \langle \bold{w},\Psi(\bold{x},\bold{h}) \rangle\f$
			 * This function has to be defined the user as it is applications specific, since
			 * it depends on the user defined latent feature and latent label.
			 *
			 * @param w weight vector
			 * @param idx index of the example
			 * @return returns \f$h^{*}\f$ for the given example
			 */
			virtual CData* infer_latent_variable(const SGVector<float64_t>& w, index_t idx)=0;

			/** Calculates \f$argmax_{h} \langle \bold{w},\Psi(\bold{x},\bold{h}) \rangle\f$
			 * The default implementaiton calculates the argmax_h only on the positive examples.
			 *
			 * @param w weight vector (cutting plane) supplied by the underlying optimizer.
			 */
			virtual void argmax_h(const SGVector<float64_t>& w);

			/** cache the PSI vectors
			 *
			 */
			void cache_psi_features();

			/** get the cached PSI vectors
			 *
			 * @return the cached PSI vectors
			 */
			CDotFeatures* get_cached_psi_features() const;

			/** get caching
			 *
			 * @return true if caching of PSI vectors is enabled; false otherwise
			 */
			inline bool get_caching() const
			{
				return m_do_caching;
			}

			/** set caching of PSI features
			 *
			 * @param caching true if one wants to cache PSI vectors; false otherwise
			 */
			inline void set_caching(bool caching)
			{
				m_do_caching = caching;
			}

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LatentModel"; }

		protected:
			/** latent features for training */
			CLatentFeatures* m_features;
			/** corresponding labels for the train set */
			CLatentLabels* m_labels;
			/** boolean that indicates whether caching of PSI vectors is enabled or not */
			bool m_do_caching;
			/** cached PSI feature vectors after argmax_h */
			CDotFeatures* m_cached_psi;

		private:
			/** register the parameters */
			void register_parameters();
	};
}

#endif /* __LATENTMODEL_H__ */

