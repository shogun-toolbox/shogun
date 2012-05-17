/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTURED_LOSS_H__
#define _STRUCTURED_LOSS_H__

#include <shogun/base/SGObject.h>
#include <shogun/features/Features.h>
#include <shogun/features/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>

namespace shogun {

class CStructuredLoss;

/**
 * @brief Abstract class StructuredLoss that represents the application
 * independent loss \f$l(x,y,w)\f$  used in structured output (SO) problems. 
 * This type of loss function must be used since the applicacion dependent 
 * \f$Delta\f$ loss defined in a StructuredModel class may be non convex or 
 * discontinuous w.r.t. \f$w\f$. This class is thought to be used as the 
 * interface for the family of \f$l\f$ functions that contains e.g. hinge loss.
 */
class CStructuredLoss : public CSGObject
{
	public:
		/** default constructor */
		CStructuredLoss();

		/** destructor */
		~CStructuredLoss();

		/** abstract method to compute the loss
		 *
		 * @param z where to evaluate the function
		 * @return the loss
		 */
		virtual float64_t compute(float64_t z) = 0;

		/** abstract method the first gradient of the loss
		 *
		 * @param z where to evaluate the gradient of the function
		 * @return the gradient (first derivative) of the loss
		 */

		virtual float64_t compute_gradient(float64_t z) = 0;

		/** abstract method the second gradient of the loss
		 *
		 * @param z where to evaluate the subgradient of the function
		 * @return the subgradient (second derivative or Hessian) of the 
		 *		loss
		 */
		virtual float64_t compute_subgradient(float64_t z) = 0;

		/** is smooth? 
		 *
		 * @return is_smooth whether the loss function is smooth
		 */
		inline bool is_smooth() { return m_is_smooth; }

		/** is convex?
		 *
		 * @return is_convex whether the loss function is convex
		 */
		inline bool is_convex() { return m_is_convex; }

		/** is positive?
		 *
		 * @return is_positive whether the loss function is positive
		 */
		inline bool is_positive() { return m_is_positive; }

		/** @return name of SGSerializable */
		inline virtual const char* get_name() const 
			{ return "StructuredLoss"; }
			
	protected:
		/** whether the loss function is smooth */
		bool m_is_smooth;

		/** whether the loss function is convex */
		bool m_is_convex;

		/** whether the loss function is positive, 
		 *  i.e. \f$ l: (\mathcal{X, Y}, \Re^{n}) \to \Re^{+}? \f$ 
		 */
		bool m_is_positive;

}; /* class CStructuredLoss */

} /* namespace shogun */

#endif /* _STRUCTURED_LOSS_H__ */
