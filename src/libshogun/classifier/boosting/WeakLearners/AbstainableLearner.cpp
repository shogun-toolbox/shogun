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


#include <cassert>
#include <limits> // for numeric_limits<>
#include <cmath>

#include "classifier/boosting/StrongLearners/AdaBoostMHLearner.h"
#include "classifier/boosting/WeakLearners/AbstainableLearner.h"

#include "classifier/boosting/Utils/Utils.h"
#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/IO/SortedData.h"

namespace MultiBoost {

// ------------------------------------------------------------------------------

void AbstainableLearner::declareArguments(nor_utils::Args& args)
{
   BaseLearner::declareArguments(args);

   args.declareArgument("abstention", 
                        "Activate the abstention. Available types are:\n"
                        "  greedy: sorting and checking in O(k^2)\n"
                        "  full: the O(2^k) full search\n"
                        "  real: use the AdaBoost.MH with real valued predictions\n"
                        "  classwise: abstain if classwise edge <= theta",
                        1, "<type>");
}

// ------------------------------------------------------------------------------

void AbstainableLearner::initLearningOptions(const nor_utils::Args& args)
{
   BaseLearner::initLearningOptions(args);

   // set abstention
   if ( args.hasArgument("abstention") )
   {
      string abstType = args.getValue<string>("abstention", 0);

      if (abstType == "greedy")
         _abstention = ABST_GREEDY;
      else if (abstType == "full")
         _abstention = ABST_FULL;
      else if (abstType == "real")
         _abstention = ABST_REAL;
      else if (abstType == "classwise")
         _abstention = ABST_CLASSWISE;
      else
      {
         cerr << "ERROR: Invalid type of abstention <" << abstType << ">!!" << endl;
         exit(1);
      }
   }
}

// ------------------------------------------------------------------------------
/*
float AbstainableLearner::classify(InputData* pData, int idx, int classIdx)
{
   return _v[classIdx] * phi( pData, idx, classIdx );
}
*/
// -----------------------------------------------------------------------

void AbstainableLearner::save(ofstream& outputStream, int numTabs)
{
   // Calling the super-class method
   BaseLearner::save(outputStream, numTabs);

   // save the vote vector
   outputStream << Serialization::vectorTag("vArray", _v, 
					    _pTrainingData->getClassMap(), 
					    "class", (float) 0.0, numTabs) << endl;
}

// -----------------------------------------------------------------------

void AbstainableLearner::subCopyState(BaseLearner *pBaseLearner)
{
   BaseLearner::subCopyState(pBaseLearner);

   AbstainableLearner* pAbstainableLearner =
      dynamic_cast<AbstainableLearner*>(pBaseLearner);

   pAbstainableLearner->_v = _v;
   pAbstainableLearner->_abstention = _abstention;
}

// -----------------------------------------------------------------------

void AbstainableLearner::load(nor_utils::StreamTokenizer& st)
{
   // Calling the super-class method
   BaseLearner::load(st);

   // load vArray data
   UnSerialization::seekAndParseVectorTag(st, "vArray", _pTrainingData->getClassMap(), 
					  "class", _v);
}

// ------------------------------------------------------------------------------

float AbstainableLearner::getEnergy(vector<sRates>& mu, float& alpha, vector<float>& v)
{
   const int numClasses = mu.size();

   sRates eps;

   // Get the overall error and correct rates
   for (int l = 0; l < numClasses; ++l)
   {
      eps.rMin += mu[l].rMin;
      eps.rPls += mu[l].rPls;
   }

   //if( eps.rMin < 0 ) eps.rMin = -eps.rMin;
   // assert: eps- + eps+ + eps0 = 1
//    assert( eps.rMin + eps.rPls <= 1 + _smallVal &&
//            eps.rMin + eps.rPls >= 1 - _smallVal);

//    cout << "eps = " << eps.rMin << " " << eps.rPls;
   float currEnergy = 0;
   if ( _abstention != ABST_REAL )
   {
      if ( nor_utils::is_zero(_theta) )
      {
         alpha = getAlpha(eps.rMin, eps.rPls);
         currEnergy = BaseLearner::getEnergy( eps.rMin, eps.rPls );
      }
      else
      {
         alpha = getAlpha(eps.rMin, eps.rPls, _theta);
         currEnergy = BaseLearner::getEnergy( eps.rMin, eps.rPls, alpha, _theta );
      }
   }

   // perform abstention
   switch(_abstention)
   {
      case ABST_GREEDY:
         // alpha and v are updated!
         currEnergy = doGreedyAbstention(mu, currEnergy, eps, alpha, v);
         break;
      case ABST_FULL: 
         // alpha and v are updated!
         currEnergy = doFullAbstention(mu, currEnergy, eps, alpha, v);
         break;
      case ABST_REAL:
         // alpha and v are updated!
         currEnergy = doRealAbstention(mu, eps, alpha, v);
         break;
      case ABST_CLASSWISE:
         // alpha and v are updated!
         currEnergy = doClasswiseAbstention(mu, eps, alpha, v);
         break;
      case ABST_NO_ABSTENTION:
         break;
   }

   // Condition: eps_pls > eps_min + theta equivalent to alpha < 0!!
   if (alpha < 0)
	   currEnergy = std::numeric_limits<float>::max();

   return currEnergy; // this is what we are trying to minimize: 2*sqrt(eps+*eps-)+eps0
}

// -----------------------------------------------------------------------

float AbstainableLearner::doGreedyAbstention(vector<sRates>& mu, float currEnergy, 
                                        sRates& eps, float& alpha, vector<float>& v)
{
   // Abstention is performed by evaluating the class-wise error
   // and the case in which one element (the one with the highest mu_pls * mu_min value)
   // is ignored, that is has v[el] = 0

   const int numClasses = mu.size();

   // Sorting the energies for each vote
   sort(mu.begin(), mu.end());

   bool changed;
   sRates newEps;
   float newAlpha;
   float newEnergy;

   do
   {
      changed = false;

      for (int l = 0; l < numClasses; ++l)
      {
         if ( v[ mu[l].classIdx ] != 0 ) 
         {
            newEps.rMin = eps.rMin - mu[l].rMin;
            newEps.rPls = eps.rPls - mu[l].rPls;
            newEps.rZero = eps.rZero + mu[l].rZero;

            if ( nor_utils::is_zero(_theta) )
            {
               newAlpha = getAlpha(newEps.rMin, newEps.rPls);
               newEnergy = BaseLearner::getEnergy(newEps.rMin, newEps.rPls);
            }
            else
            {
               newAlpha = getAlpha(newEps.rMin, newEps.rPls, _theta);
               newEnergy = BaseLearner::getEnergy(newEps.rMin, newEps.rPls, newAlpha, _theta);
            }

            if ( newAlpha > 0 && newEnergy + _smallVal < currEnergy )
            {
               // ok, this is v = 0!!
               changed = true;

               currEnergy = newEnergy;
               eps = newEps;

               v[ mu[l].classIdx ] = 0;
               alpha = newAlpha;

               // assert: eps- + eps+ + eps0 = 1
               assert( eps.rMin + eps.rPls + eps.rZero <= 1 + _smallVal &&
                       eps.rMin + eps.rPls + eps.rZero >= 1 - _smallVal );
            }
         } // if
      } //for

   } while (changed);

   return currEnergy;
}

// -----------------------------------------------------------------------

float AbstainableLearner::doFullAbstention(const vector<sRates>& mu, float currEnergy, 
                                      sRates& eps, float& alpha, vector<float>& v)
{
   const int numClasses = mu.size();

   vector<char> best(numClasses, 1);
   vector<char> candidate(numClasses);
   sRates newEps; // candidate
   float newAlpha;
   float newEnergy;

   sRates bestEps = eps;

   for (int l = 1; l < numClasses; ++l)
   {
      // starts with an array with just one 0 (and the rest 1), 
      // then two 0, then three 0, etc..
      fill( candidate.begin(), candidate.begin()+l, 0 );
      fill( candidate.begin()+l, candidate.end(), 1 );

      // checks all the possible permutations of such array
      do {

         newEps = eps;

         for ( int j = 0; j < numClasses; ++j )
         {
            if ( candidate[j] == 0 )
            {
               newEps.rMin -= mu[j].rMin;
               newEps.rPls -= mu[j].rPls;
               newEps.rZero += mu[j].rZero;
            }
         }

         if ( nor_utils::is_zero(_theta) )
         {
            newAlpha = getAlpha(newEps.rMin, newEps.rPls);
	    newEnergy = BaseLearner::getEnergy(newEps.rMin, newEps.rPls);
         }
         else
         {
            newAlpha = getAlpha(newEps.rMin, newEps.rPls, _theta);
	    newEnergy = BaseLearner::getEnergy(newEps.rMin, newEps.rPls, newAlpha, _theta);
         }

         if ( newAlpha > 0 && newEnergy + _smallVal < currEnergy )
         {
            currEnergy = newEnergy;

            best = candidate;
            alpha = newAlpha;
            bestEps = newEps;

            // assert: eps- + eps+ + eps0 = 1
            assert( newEps.rMin + newEps.rPls + newEps.rZero <= 1 + _smallVal &&
                    newEps.rMin + newEps.rPls + newEps.rZero >= 1 - _smallVal );
         }

      } while ( next_permutation(candidate.begin(), candidate.end()) );

   }

   for (int l = 0; l < numClasses; ++l)
      v[l] = v[l] * best[l]; // avoiding v[l] *= best[l] because of a (weird) warning

   eps = bestEps;

   return currEnergy; // this is what we are trying to minimize: 2*sqrt(eps+*eps-)+eps0
}

// -----------------------------------------------------------------------

float AbstainableLearner::doRealAbstention( const vector<sRates>& mu, const sRates& eps,
                                             float& alpha, vector<float>& v)
{
   const int numClasses = mu.size();

   float currEnergy = 0;
   alpha = 1; // setting alpha to 1

   if ( nor_utils::is_zero(_theta) )
   {
      for (int l = 0; l < numClasses; ++l)
      {
         v[l] *= getAlpha(mu[l].rMin, mu[l].rPls);
         currEnergy += sqrt( mu[l].rMin * mu[l].rPls );
      }

      currEnergy *= 2;
   }
   else
   {
      float alpha_l;
      for (int l = 0; l < numClasses; ++l)
      {
	 alpha_l = getAlpha(mu[l].rMin, mu[l].rPls, _theta);
	 if (alpha_l <= 0)
	    alpha_l = 0;

	 v[l] *= alpha_l;

	 currEnergy += exp( _theta * alpha_l ) *
                        ( mu[l].rPls * exp(-alpha_l) + 
                          mu[l].rMin * exp(alpha_l) ); // =  mu[l].rZero if alpha_l = 0
         }

   }


   return currEnergy;
}

// -----------------------------------------------------------------------

float AbstainableLearner::doClasswiseAbstention(vector<sRates>& mu, 
						 sRates& eps, float& alpha, vector<float>& v)
{
   // Abstention is performed by evaluating the class-wise error
   // if mu.rPls - mu.rMin < _theta then abstain on the class
   // only makes sense to use with _theta > 0-

   const int numClasses = mu.size();

   for (int l = 0; l < numClasses; ++l)
   {
       if ( mu[l].rPls - mu[l].rMin < _theta )
       {
	   eps.rMin -= mu[l].rMin;
	   eps.rPls -= mu[l].rPls;
	   eps.rZero += mu[l].rMin + mu[l].rPls;
	   v[l] = 0;
       }   
   }
 
   alpha = getAlpha(eps.rMin, eps.rPls, _theta);
   return BaseLearner::getEnergy(eps.rMin, eps.rPls, alpha, _theta);

}

} // end of namespace MultiBoost
