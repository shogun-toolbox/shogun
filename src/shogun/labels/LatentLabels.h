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
	/** @brief class LatentData used to store information about latent
	 * variables.
	 * TODO
	 */
	class CLatentData : public CSGObject
	{
		public:
			/** constructor */
			CLatentData();

			/** destructor */
			virtual ~CLatentData();

			/** get name */
			virtual const char* get_name() const { return "LatentData"; }
	};

	/** @brief class LatentLabels used to store latent labels 
	 * TODO
	 */
	class CLatentLabels : public CBinaryLabels
	{
		public:
			/** constructor */
			CLatentLabels();

			/** constructor
			 *
			 * @param num_labels number of labels
			 */
			CLatentLabels(int32_t num_labels);

			/** destructor */
			virtual ~CLatentLabels();

			/** get labels */
			CDynamicObjectArray* get_labels() const;

			/** get latent label
			 * 
			 * @param idx index of label
			 */
			CLatentData* get_latent_label(int32_t idx);

			/** add latent label
			 *
			 * @param label label to add
			 */
			void add_latent_label(CLatentData* label);

			/** set latend label
			 *
			 * @param idx index of latent label
			 * @param label value of latent label
			 */
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

			/** get name */
			virtual const char* get_name() const { return "LatentLabels"; }

		protected:
			/** the vector of labels */
			CDynamicObjectArray* m_latent_labels;

		private:
			void init();
	};
}

#endif /* __LATENTLABELS_H__ */

