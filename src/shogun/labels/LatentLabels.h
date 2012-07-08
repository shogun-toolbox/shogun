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
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{
	class CLatentData : public CSGObject
	{
		public:
			CLatentData();

			virtual ~CLatentData();

			virtual const char* get_name() const { return "LatentData"; }
	};

	class CLatentLabels : public CBinaryLabels
	{
		public:
			CLatentLabels();

			CLatentLabels(int32_t num_labels);

			virtual ~CLatentLabels();

			CDynamicObjectArray* get_labels() const;

			CLatentData* get_latent_label(int32_t idx);

			void add_latent_label(CLatentData* label);

			bool set_latent_label(int32_t idx, CLatentData* label);

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

			virtual const char* get_name() const { return "LatentLabels"; }

		protected:
			/** the vector of labels */
			CDynamicObjectArray* m_latent_labels;

		private:
			void init();
	};
}

#endif /* __LATENTLABELS_H__ */

