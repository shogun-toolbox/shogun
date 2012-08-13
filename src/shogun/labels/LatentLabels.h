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

#include <shogun/labels/BinaryLabels.h>
#include <shogun/lib/Data.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{
	/** @brief abstract class for latent labels
	 * As latent labels always depends on the given application, this class
	 * only defines the API that the user has to implement for latent labels.
	 */
	class CLatentLabels : public CBinaryLabels
	{
		public:
			/** default ctor */
			CLatentLabels();

			/** constructor
			 *
			 * @param num_samples the number of labels
			 */
			CLatentLabels(int32_t num_labels);

			/** destructor */
			virtual ~CLatentLabels();

			/** get all the stored latent labels
			 *
			 * @return the CDynamicObjectArray with the latent labels in it
			 */
			CDynamicObjectArray* get_labels() const;

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
			virtual ELabelType get_label_type() { return LT_LATENT; }

			/** helper method used to specialize a base class instance
			 *
			 * @param base_labels its dynamic type must be CLatentLabels
			 */
			static CLatentLabels* obtain_from_generic(CLabels* base_labels);

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LatentLabels"; }

		protected:
			/** the vector of labels */
			CDynamicObjectArray* m_latent_labels;

		private:
			/** initalize the values to default values */
			void init();
	};
}

#endif /* __LATENTLABELS_H__ */

