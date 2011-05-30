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

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/classifier/PluginEstimate.h>
#include <shogun/features/Labels.h>

namespace shogun
{
class CSGInterface;

class CGUIPluginEstimate : public CSGObject
{

	public:
		CGUIPluginEstimate(CSGInterface* interface);
		~CGUIPluginEstimate();

		/** create new estimator */
		bool new_estimator(float64_t pos, float64_t neg);
		/** train estimator */
		bool train();
		bool marginalized_train(char* param);
		/** test estimator */
		bool load(char* param);
		bool save(char* param);

		inline CPluginEstimate* get_estimator() { return estimator; }

		CLabels* apply();
		float64_t apply(int32_t idx);

		/** @return object name */
		inline virtual const char* get_name() const { return "GUIPluginEstimate"; }

	protected:
		CSGInterface* ui;

		CPluginEstimate* estimator;
		float64_t pos_pseudo;
		float64_t neg_pseudo;
};
}
#endif
