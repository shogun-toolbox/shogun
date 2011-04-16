/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Tanmoy Mukherjee
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2011 IIIT Hyderabad
 */

#include "lib/config.h"

#ifndef BESSELKERNEL_H_
#define BESSELKERNEL_H_

#include "lib/common.h"
#include "kernel/Kernel.h"
#include "distance/Distance.h"

namespace shogun { 

class CDistance;
/**  Bessel kernel
 *
 *  It is described as
 *
 * \f[
 * 		K(x,x') = -Bessel^{n}_{nu+1}{1}{\sigma*\|x-x'\|^2}
 * \f]
 *
 */


class CBesselKernel: public CKernel
{
  
   public:
    /** Default Constructor */      
   CBesselKernel();
   /** constructor
	 * 
	 * @param size cache size
	 * @param sigma kernel parameter sigma
	 * @param dist distance
	 */   

   CBesselKernel(int32_t size, float64_t sigma, CDistance* dist);
   /** initialize kernel with features
	 * 
	 * @param l features of left-side
	 * @param r features of right-side
	 * @return true if successful
	 */

  CBesselKernel(CFeatures *l, CFeatures *r, float64_t coef, CDistance* dist);
  /** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
 
   virtual bool init(CFeatures* l, CFeatures* r);
   

  

   inline virtual EKernelType get_kernel_type() { return K_BESSEL; }

   /**
   * @return type of features
   */
   inline virtual EFeatureType get_feature_type() { return distance->get_feature_type(); } 
  /**
  * @return class of features
  */
  
   inline virtual EFeatureClass get_feature_class() { return distance->get_feature_class(); } 
   
   /**
   * @return name of kernel
   */
   inline virtual const char* get_name() const { return "Bessel"; }
   
   inline virtual void set_sigma(float64_t s)
	{
	  	sigma=s;
	}

   virtual ~CBesselKernel();

   
   protected:
   /// distance to be used
	CDistance* distance;

	/// sigma parameter of kernel
	float64_t sigma;

	/**
	 * compute kernel for specific feature vectors
	 * corresponding to [idx_a] of left-side and [idx_b] of right-side
	 * @param idx_a left-side index
	 * @param idx_b right-side index
	 * @return kernel value
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);

        /** Can (optionally) be overridden to post-initialize some
	 *  member variables which are not PARAMETER::ADD'ed.  Make
	 *  sure that at first the overridden method
	 *  BASE_CLASS::LOAD_SERIALIZABLE_POST is called.
	 *
	 *  @exception ShogunException Will be thrown if an error
	 *                             occurres.
	 */
	virtual void load_serializable_post(void) throw (ShogunException);
  
    private:
          void init();

};
}    
#endif /* BESSELKERNEL_H_ */
