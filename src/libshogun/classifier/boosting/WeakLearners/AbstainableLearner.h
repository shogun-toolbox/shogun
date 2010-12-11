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


/**
* \file AbstainableLearner.h It represents all weak learners that explicitly
* use a vote vector, that is, implement h(x,l) as v[l] * phi(x,l). Vote vector 
* operations (like abstaining) is implemented here. Normally most of the weak 
* learners of this type are of the form v[l] * phi(x) (which could be called 
* "ClasswiseLearner"), except for MultiStumpLearner.
*/
#pragma warning( disable : 4786 )

#ifndef __ABSTAINABLE_LEARNER_H
#define __ABSTAINABLE_LEARNER_H

#include "classifier/boosting/WeakLearners/BaseLearner.h"
#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/Others/Rates.h"

#include <vector>
#include <fstream>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
* \file AbstainableLearner.h It represents all weak learners that explicitly
* use a vote vector, that is, implement h(x,l) as v[l] * phi(x,l). Vote vector 
* operations (like abstaining) is implemented here.
*
* \date 19/09/06
*/
class AbstainableLearner : public BaseLearner
{
public:
   vector<float> _v;

   /**
   * The constructor. It initializes theta to zero and the abstention to false
   * (that is, their default values).
   * \date 19/107/2006
   */
   AbstainableLearner() 
      : _abstention(ABST_NO_ABSTENTION) {}

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~AbstainableLearner() {}

   /**
   * Declare weak-learner-specific arguments.
   * These arguments will be added to the list of arguments under 
   * the group specific of the weak learner. It is called
   * automatically in main, when the list of arguments is built up.
   * Use this method to declare the arguments that belongs to
   * the weak learner only.
   * 
   * This class declares the argument "abstention" and "edgeoffset".
   * \param args The Args class reference which can be used to declare
   * additional arguments.
   * \date 19/07/2006
   */
   virtual void declareArguments(nor_utils::Args& args);

   /**
   * Set the arguments of the algorithm using the standard interface
   * of the arguments. Call this to set the arguments asked by the user.
   * \param args The arguments defined by the user in the command line.
   * \date 19/07/2006
   */
   virtual void initLearningOptions(const nor_utils::Args& args);

   /**
   * Return {+1, -1} for the given class and value using the learned classifier.
   * \param pData The pointer to the data
   * \param idx The index of the example to classify
   * \param classIdx The index of the class
   * \remark Passing the data and the index to the example is not nice at all.
   * This will soon be replace with the passing of the example itself in some
   * form (probably a structure to the example).
   * \return +1 if the classifier thinks that \a val belongs to class 
   * \a classIdx, -1 if it does not and 0 if it abstain. If ABST_REAL is selected
   * the value returned is a range between -1 and +1 which holds the confidence
   * in the classification.
   * \date 13/11/2005
   */
   virtual float classify(InputData* pData, int idx, int classIdx)
	{
	   return _v[classIdx] * phi( pData, idx, classIdx );
	}


   /**
   * Save the current object information needed for classification,
   * that is: \a _v, The alignment vector and \a _selectedColumn, the column of the 
   * data with that yielded the lowest error
   * \param outputStream The stream where the data will be saved
   * \param numTabs The number of tabs before the tag. Useful for indentation
   * \remark To fully save the object it is \b very \b important to call
   * also the super-class method.
   * \see BaseLearner::save()
   * \date 19/07/2006
   */
   virtual void save(ofstream& outputStream, int numTabs = 0);

   /**
   * Load the xml file that contains the serialized information
   * needed for the classification and that belongs to this class
   * \param st The stream tokenizer that returns tags and values as tokens
   * \see save()
   * \date 19/07/2006
   */
   virtual void load(nor_utils::StreamTokenizer& st);

   /**
   * Copy all the info we need in classify().
   * pBaseLearner was created by subCreate so it has the correct (sub) type.
   * Usually one must copy the same fields that are loaded and saved. Don't 
   * forget to call the parent's subCopyState().
   * \param pBaseLearner The sub type pointer into which we copy.
   * \see save
   * \see load
   * \see classify
   * \see ProductLearner::run()
   * \date 25/05/2007
   */
   virtual void subCopyState(BaseLearner *pBaseLearner);

protected:

   /**
   * A discriminative function. 
   * \param pData The data
   * \param idx The index of the data point
   * \param classIdx The class index
   * \date 19/07/2006
   */
   virtual float phi(InputData* pData, int idx, int classIdx) const = 0;

   /**
   * Return the energy of the current learner. The energy is defined as
   * \f[ 
   * Z = 2 \sqrt{\epsilon_+ \epsilon_-} + \epsilon_0
   * \f]
   * and it is the value to minimize.
   * \param alpha The value of alpha that will be updated by this function minimizing the
   * energy and using the helper function provided by BaseLearner.
   * \param mu The class rates.
   * \param v The alignment vector that will be updated in the case of abstention.
   * \return The energy value that we want minimize.
   * \date 12/11/2005
   */
   virtual float getEnergy(vector<sRates>& mu, float& alpha, vector<float>& v);

   /**
   * Updates the \a v vector (alignment vector) using a greedy abstention algorithm.
   * We do not leave the decision to the weak learner as usual, but we add 0 to the 
   * decisions of the alignment vector \a v. This is done by optimizing the energy value 
   * with a greedy algorithm (that, for the time being, is not proved to be optimal. 
   * We first get \a v from one of the stump algorithms (see for instance 
   * SingleStumpLearner::findThreshold()). 
   * Then, in an iteration over the classes, we select the "best" element of v to set 
   * to 0, that is, the one that decreases the energy the most.
   * \param mu The class rates. It is not \a const because sort() is called.
   * \param currEnergy The current energy value, obtained in getEnergy().
   * \param eps The current epsilons, that the overall error, correct and abstention rates.
   * \param alpha The value of alpha that will be updated by this function minimizing the
   * energy and using the helper function provided by BaseLearner.
   * \param v The alignment vector that will be updated in the case of abstention.
   * \remark The complexity of this algorithm is O(k^2).
   * \date 28/11/2005
   */
   virtual float doGreedyAbstention(vector<sRates>& mu,  float currEnergy, 
                                     sRates& eps, float& alpha, vector<float>& v);

   /**
   * Updates the \a v vector (alignment vector) evaluating all the possible combinations.
   * Again we do not leave the abstention to the weak learner but we add 0 to the
   * alignment vector \a v, which decreases most the energy function.
   * \param mu The class rates.
   * \param currEnergy The current energy value, obtained in getEnergy().
   * \param eps The current epsilons, that is the overall error, correct and abstention rates.
   * \param alpha The value of alpha that will be updated by this function minimizing the
   * energy and using the helper function provided by BaseLearner.
   * \param v The alignment vector that will be updated in the case of abstention.
   * \remark The complexity of this algorithm is O(2^k).
   * \date 28/11/2005
   */
   virtual float doFullAbstention(const vector<sRates>& mu,  float currEnergy, 
                                   sRates& eps, float& alpha, vector<float>& v);

   /**
   * Update the \a v vector (alignment vector) with \b real values, setting alpha to 1.
   * This corresponds to "AdaBoost.MH with real valued predictions" (see "BoosTexter:
   * A Boosting-based System for Text Categorization" by Schapire & Singer), which must 
   * \b not be confused with Real AdaBoost, where the weak learner returns a confidence 
   * on its prediction. 
   * \param mu The class rates.
   * \param eps The current epsilons, that is the overall error, correct and abstention rates.
   * \param alpha The value of alpha which will be turned into 1.
   * \param v The alignment vector that will be updated with the real valued values.
   * \remark Because here we have an alpha which is not uniform among all the classes, (\a v in
   * fact becomes the alpha) the energy function to minimize becomes:
   * \f[
   * Z = \sum_{i=1}^n \sum_{\ell=1}^k w_{i,\ell} \exp{ \left( -\alpha_\ell h_\ell(x_i) y_{i,\ell} \right) },
   * \f]
   * which simplifies to the new energy function that we have to minimize:
   * \f[
   * Z = 2 \sum_{\ell=1}^k \sqrt{ \mu_{-,\ell} \mu_{+,\ell} }
   * \f]
   * \remark If a margin has been selected (\f$\theta > 0\f$), then the energy function
   * is computed as usual.
   * \see sRates
   * \see doGreedyAbstention
   * \see doFullAbstention
   * \date 10/2/2006
   * \bug The actual formula should be
   * \f[
     Z = \mu_0 + 2 \sum_{\ell=1}^k \sqrt{ \mu_{-,\ell} \mu_{+,\ell} },
   * \f]
   * but we do not consider the abstention at weak learner level yet.
   */
   virtual float doRealAbstention(const vector<sRates>& mu, const sRates& eps,
                                   float& alpha, vector<float>& v);

   /**
   * Updates the \a v vector (alignment vector) using a classwise abstention algorithm.
   * If mu.rPls - mu.rMin < _theta then abstain on the class.
   * Only makes sense to use with _theta > 0.
   * Similar to what real abstention does, but keeps +-1 votes.
   * \param mu The class rates. It is not \a const because sort() is called.
   * \param currEnergy The current energy value, obtained in getEnergy().
   * \param eps The current epsilons, that the overall error, correct and abstention rates.
   * \param alpha The value of alpha that will be updated by this function minimizing the
   * energy and using the helper function provided by BaseLearner.
   * \param v The alignment vector that will be updated in the case of abstention.
   * \remark The complexity of this algorithm is O(k^2).
   * \date 27/04/2007
   */
   virtual float doClasswiseAbstention(vector<sRates>& mu,  
					sRates& eps, float& alpha, vector<float>& v);

   /**
   * The class-wise abstention/alignment vector. 
   * If the abstention is \b not set to ABST_REAL, it is obtained simply with
   * \f[ 
   *   v_\ell = \begin{cases}
   *   +1 & \mbox{ if } \mu_{\ell+} > \mu_{\ell-}\\
   *   -1 & \mbox{ otherwise.}
   *   \end{cases} 
   * \f]
   * where  \f$\mu\f$ are defined in sRates.
   * 
   * In the case of ABST_REAL, this vector becomes the class-wise alpha value,
   * and alpha is set to 1.
   * \see sRates
   * \see doRealAbstention
   * \see eAbstType
   * \date 11/11/2005
   */

   /**
   * The type of abstention.
   * \date 28/11/2005
   * \see doGreedyAbstention
   * \see doFullAbstention
   * \see doRealAbstention
   */
   enum eAbstType
   {
      ABST_NO_ABSTENTION, //!< No abstention is performed. 
      ABST_GREEDY, //!< The abstention is type greedy, which complexity is O(k^2), where k is the number of classes.
      ABST_REAL, //!< The value of the v vector is float instead of {-1,0,+1} as described in Boostexter's paper
      ABST_FULL, //!< The abstention is full, which complexity is O(2^k).
      ABST_CLASSWISE  //! The value of the v vector is {-1,0,+1} abstain if classwise edge < theta
   };
   eAbstType      _abstention; //!< Activate abstention. Default = 0 (no abstention);
};

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

} // end of namespace MultiBoost

#endif // __ABSTAINABLE_LEARNER_H
