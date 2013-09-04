/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 * ported from scikit-learn 
 */

#ifndef FASTICA_H_
#define FASTICA_H_

#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#include <shogun/converter/ica/ICAConverter.h>
#include <shogun/features/Features.h>

namespace shogun
{

class CFeatures;

/** @brief class FastICA
 * 
 * Implements the FastICA (Independent 
 * Component Analysis) algorithm
 * 
 * A. Hyvarinen and E. Oja, Independent Component Analysis:
 * Algorithms and Applications, Neural Networks, 13(4-5), 2000,
 * pp. 411-430`
 */
class CFastICA: public CICAConverter
{
	public:
		
		/** constructor */
		CFastICA();

		/** destructor */
		virtual ~CFastICA();

		/** apply to features
		 * @param features features to embed
		 */
		virtual CFeatures* apply(CFeatures* features);

		/** setter for whiten flag
		 * @param whiten
		 */
		void set_whiten(bool whiten);

		/** getter for whiten flag
		 * @return whiten
		 */
		bool get_whiten() const;

		/** @return object name */
		virtual const char* get_name() const { return "FastICA"; };

	protected:

		/** init */
		void init();

	private:
		
		/** whiten */
		bool whiten;
	
};	
}
#endif // HAVE_EIGEN3
#endif // FASTICA
