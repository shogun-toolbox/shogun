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
			 */
			CLatentModel(CLatentFeatures* feats, CLatentLabels* labels);

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

			/** Calculate the PSI vector for a given sample
			 *
			 * @param idx index of the sample
			 * @return PSI vector
			 */
			virtual SGVector<float64_t> get_psi_feature_vector(index_t idx)=0;

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

		private:
			/** register the parameters */
			void register_parameters();
	};
}

#endif /* __LATENTMODEL_H__ */

