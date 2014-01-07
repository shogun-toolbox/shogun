/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef _DIRECTOR_LATENT_MODEL_H_
#define _DIRECTOR_LATENT_MODEL_H_

#include <latent/LatentModel.h>
#include <lib/config.h>
#ifdef USE_SWIG_DIRECTORS
namespace shogun
{

class CLatentModel;

#define IGNORE_IN_CLASSLIST
/**
 * @brief Class CDirectorLatentModel that represents the application specific model
 * with latent variable svm in target interface language. It is a base class
 * that needs to be extended with real implementations before using.
 *
 * @see CLatentModel
 */
IGNORE_IN_CLASSLIST class CDirectorLatentModel : public CLatentModel
{
	public:
		/** default constructor */
		CDirectorLatentModel();

		/** destructor */
		virtual ~CDirectorLatentModel();

		/**
		 * return the dimensionality of the joint feature space, i.e.
		 * the dimension of the weight vector \f$w\f$
		 */
		virtual int32_t get_dim() const;

		/** Calculate the PSI vectors for all features
		 *
		 * @return PSI vectors
		 */
		virtual CDotFeatures* get_psi_feature_vectors();

		/** User defined \f$h^{*} = argmax_{h} \langle \bold{w},\Psi(\bold{x},\bold{h}) \rangle\f$
		 * This function has to be defined the user as it is applications specific, since
		 * it depends on the user defined latent feature and latent label.
		 *
		 * @param w weight vector
		 * @param idx index of the example
		 * @return returns \f$h^{*}\f$ for the given example
		 */
		virtual CData* infer_latent_variable(const SGVector<float64_t>& w, index_t idx);

		/** Calculates \f$argmax_{h} \langle \bold{w},\Psi(\bold{x},\bold{h}) \rangle\f$
		 * The default implementaiton calculates the argmax_h only on the positive examples.
		 *
		 * @param w weight vector (cutting plane) supplied by the underlying optimizer.
		 */
		virtual void argmax_h(const SGVector<float64_t>& w);

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "DirectorLatentModel"; }

}; /* class CDirectorLatentModel */
} /* namespace shogun */
#endif /* USE_SWIG_DIRECTORS */
#endif /* _DIRECTOR_LATENT_MODEL_H_ */

