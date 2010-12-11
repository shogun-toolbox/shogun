/*
*
*    MultiBoost - Multi-purpose boosting package
*
*    Copyright (C) 2010   AppStat group
*                         Laboratoire de l'Accelerateur Lineaire
*                         Universite Paris-Sud, 11, CNRS
*
*    This file is part of the MultiBoost library
*
*    This library is free software; you can redistribute it 
*    and/or modify it under the terms of the GNU General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    General Public License for more details.
*
*    You should have received a copy of the GNU General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
*
*    Contact: Balazs Kegl (balazs.kegl@gmail.com)
*             Norman Casagrande (nova77@gmail.com)
*             Robert Busa-Fekete (busarobi@gmail.com)
*
*    For more information and up-to-date version, please visit
*        
*                       http://www.multiboost.org/
*
*/


#ifndef __PER_CLASS_RATES_H
#define __PER_CLASS_RATES_H

#include <stdio.h>
#include <string.h> 

namespace MultiBoost {

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

/**
* The per class rates. 
* It is defined (using \f$\mu\f$ as variable) as
* \f{eqnarray*}
*   \mu_{\ell+} & = & \sum_{correct} w_{i,\ell}^{(t)},\\
*   \mu_{\ell-} &=& \sum_{incorrect} w_{i,\ell}^{(t)}, \\ 
*   \mu_{\ell0} & = & \mu_{\ell-} + \mu_{\ell+}
* \f}
* \remark The implemented decision stump algorithms do not abstain themselves.
* The abstention value, that is for \f$v=0\f$ is computed in getEnergy()
* \date 11/11/2005
*/
struct sRates
{
   sRates() : classIdx(-1), rPls(0), rMin(0), rZero(0) {} //!< The constructor. 
   void clear() { memset( this, 0, sizeof(sRates) ); /* classIdx = 0; rPls = 0; rMin = 0; rZero = 0;*/  }

   int     classIdx; //!< the index of the class. Needed because we will sort the vector that contains this object

   float  rPls; //!< positive rate, or \f$\mu_{\ell+}\f$
   float  rMin; //!<  negative rate, or \f$\mu_{\ell-}\f$
   float  rZero; //!< abstention rate, or \f$\mu_{\ell0}\f$

   /**
   * Overloading of the operator to allow sorting.
   * \param el The other element to be compared with
   * \date 11/11/2005
   */ 
   bool operator<(const sRates& el) const
   {
      return el.rPls * el.rMin < this->rPls * this->rMin;
   }
};


} // end of namespace MultiBoost

#endif // __PER_CLASS_RATES_H
