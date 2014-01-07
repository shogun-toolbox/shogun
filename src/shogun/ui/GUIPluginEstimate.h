/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GUIPLUGINESTIMATE_H__
#define _GUIPLUGINESTIMATE_H__

#include <lib/config.h>
#include <base/SGObject.h>
#include <classifier/PluginEstimate.h>
#include <labels/Labels.h>

namespace shogun
{
class CSGInterface;

/** @brief UI estimate */
class CGUIPluginEstimate : public CSGObject
{

	public:
		/** constructor */
		CGUIPluginEstimate();
		/** constructor
		 * @param interface
		 */
		CGUIPluginEstimate(CSGInterface* interface);
		/** destructor */
		~CGUIPluginEstimate();

		/** create new estimator */
		bool new_estimator(float64_t pos, float64_t neg);
		/** train estimator */
		bool train();
		/** marginalized train
		 * @param param
		 */
		bool marginalized_train(char* param);
		/** load
		 * @param param
		 */
		bool load(char* param);
		/** save
		 * @param param
		 */
		bool save(char* param);

		/** get estimator */
		inline CPluginEstimate* get_estimator() { return estimator; }

		/** apply */
		CLabels* apply();
		/** apply
		 * @param idx
		 */
		float64_t apply_one(int32_t idx);

		/** @return object name */
		virtual const char* get_name() const { return "GUIPluginEstimate"; }

	private:
		void init();

	protected:
		/** ui */
		CSGInterface* ui;
		/** estimator */
		CPluginEstimate* estimator;
		/** pos pseudo */
		float64_t pos_pseudo;
		/** neg pseudo */
		float64_t neg_pseudo;
};
}
#endif
