/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __LATENTLABELS_H__
#define __LATENTLABELS_H__

#include <shogun/lib/config.h>

#include <shogun/labels/Labels.h>
#include <shogun/lib/Data.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{
	/** @brief abstract class for latent labels
	 * As latent labels always depends on the given application, this class
	 * only defines the API that the user has to implement for latent labels.
	 */
	class CLatentLabels : public CLabels
	{
		public:
			/** default ctor */
			CLatentLabels();

			/** constructor
			 *
			 * @param num_samples the number of labels
			 */
			CLatentLabels(int32_t num_samples);

			/** constructor
			 *
			 * @param labels the (y_i) labels
			 */
			CLatentLabels(CLabels* labels);

			/** destructor */
			virtual ~CLatentLabels();

			/** get all the stored latent labels
			 *
			 * @return the CDynamicObjectArray with the latent labels in it
			 */
			CDynamicObjectArray* get_latent_labels() const;

			/** get the latent label of a given example
			 *
			 * @param idx index of the label
			 * @return the user defined latent label
			 */
			CData* get_latent_label(int32_t idx);

			/** append the latent label
			 *
			 * @param label latent label
			 */
			void add_latent_label(CData* label);

			/** set latent label at a given index
			 *
			 * @param idx position of the label
			 * @param label the latent label
			 * @return TRUE if success, FALSE otherwise
			 */
			bool set_latent_label(int32_t idx, CData* label);

			/** Make sure the label is valid, otherwise raise SG_ERROR.
			 *
			 * possible with subset
			 *
			 * @param context optional message to convey the context
			 */
			virtual void ensure_valid(const char* context=NULL);

			/** get label type
			 *
			 * @return label type (binary, multiclass, ...)
			 */
			virtual ELabelType get_label_type() const { return LT_LATENT; }

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LatentLabels"; }

			/** get the number of stored labels
			 *
			 * @return the number of labels
			 */
			virtual int32_t get_num_labels() const;

			/** set labels
			 *
			 * @param labels the labels (y_i)
			 */
			void set_labels(CLabels* labels);

			/** get the labels (y_i)
			 *
			 * @return the labels (y_i)
			 */
			CLabels* get_labels() const;

		protected:
			/** the of CData, the latent labels (h_i) */
			CDynamicObjectArray* m_latent_labels;
			/** the labels (y_i) */
			CLabels* m_labels;

		private:
			/** initalize the values to default values */
			void init();
	};
}

#endif /* __LATENTLABELS_H__ */

