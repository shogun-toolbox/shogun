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



#include <cmath> // for log

#include "HierarchicalStumpLearner.h"
#include "classifier/boosting/IO/ExtendableData.h"

//#include "IO/Serialization.h"
//#include "IO/SortedData.h"

#include <limits> // for numeric_limits<>

namespace MultiBoost {


//REGISTER_LEARNER_NAME(SingleStump, HierarchicalStumpLearner)
REGISTER_LEARNER(HierarchicalStumpLearner)

   // ------------------------------------------------------------------------------

   InputData* HierarchicalStumpLearner::createInputData()
{
   return new ExtendableData();
}

// ------------------------------------------------------------------------------

float HierarchicalStumpLearner::run()
{
   // Calling the super-class method
   SingleStumpLearner::run();

   ExtendableData* pTmpData = static_cast<ExtendableData*>(_pTrainingData);
   const int numWeakLearners = pTmpData->getNumWeakLearners();
   const int numExamples = pTmpData->getNumExamples();

   // add the new weak learner to the list
   pTmpData->addWeakLearner(this);

   // number of features to combine: for the time being it's here, it will
   // be a user parameter

   // Options:
   int randomCombination = 0;
   int numOfCombFeatures = 10; // max

   if (numWeakLearners > numOfCombFeatures) 
   {
      int numOfCombFeatures = 10;
      if (randomCombination) 
      {
         for (int t = 0; t < numOfCombFeatures; ++t) 
         {
            // Combine the current weak learner with a random one
            int randomIdx = static_cast<int>(numWeakLearners * 
               static_cast<double>(rand()) /
               static_cast<double>(RAND_MAX));
            combineTwoFeatures(pTmpData, numWeakLearners, randomIdx);
         }
      }
      else // correlated 
      {
         // Options:
         int numExamplesForCalculatingDist = numExamples/10;
         // 	 int numExamplesForCalculatingDist = numExamples;
         // double minDist = numeric_limits<double>::max();
	      double minDist = 0.5;
         int hamming = 0; // 0: mutual info, 1: hamming

         vector<int> nearestNeighborIndices;
         nearestNeighborIndices.reserve(numOfCombFeatures);
         vector<double> nearestNeighborDistances;
         nearestNeighborDistances.reserve(numOfCombFeatures);
         for (int t = 0; t < numOfCombFeatures; ++t)
         { 
            nearestNeighborIndices.push_back(-1);
            nearestNeighborDistances.push_back(minDist);
         }
         for (int j = 0; j < numWeakLearners; ++j) 
         {
            const HierarchicalStumpLearner* pWeakLearner = pTmpData->getWeakLearner(j);
            double dist = 0,d;

            if (hamming) 
            {
               for (int i = 0; i < numExamplesForCalculatingDist; ++i) 
               {
                  d = phi(pTmpData, i) - pWeakLearner->phi(pTmpData, i);
                  dist += (d > 0) ? d : -d;
               }
               dist /= (2.0 * numExamplesForCalculatingDist);
            }
            else { // mutual info
               int n11 = 0, n10 = 0, n01 = 0, n00 = 0;
               double phi1, phi2;
               for (int i = 0; i < numExamplesForCalculatingDist; ++i) 
               {
                  phi1 = phi(pTmpData, i);
                  phi2 = pWeakLearner->phi(pTmpData, i);
                  if (phi1 == 1) 
                     if (phi2 == 1)
                        ++n11;
                     else
                        ++n10;
                  else
                     if (phi2 == 1)
                        ++n01;
                     else
                        ++n00;
               }
               double m11 = mutualInformationTerm(n11,n10,n01,numExamplesForCalculatingDist);
               double m10 = mutualInformationTerm(n10,n11,n00,numExamplesForCalculatingDist);
               double m01 = mutualInformationTerm(n01,n11,n00,numExamplesForCalculatingDist);
               double m00 = mutualInformationTerm(n00,n10,n01,numExamplesForCalculatingDist);
               dist = 1 - m11 - m10 - m01 - m00;
            }
            for (int t = 0; t < numOfCombFeatures; ++t) 
            {
               if (dist < nearestNeighborDistances[t] )
               {
                  for (int t1 = numOfCombFeatures-1; t1 >= t+1; --t1)
                  {
                     nearestNeighborDistances[t1] = nearestNeighborDistances[t1-1];
                     nearestNeighborIndices[t1] = nearestNeighborIndices[t1-1];
                  }
                  nearestNeighborDistances[t] = dist;
                  nearestNeighborIndices[t] = j;
                  break;
               }
            }
         }
         for (int t = 0; t < numOfCombFeatures; ++t)
         {
            if (nearestNeighborIndices[t] != -1) 
            {
               cout << "d(" << numWeakLearners << "," << nearestNeighborIndices[t] <<
                  ") = " << nearestNeighborDistances[t] << endl;
               combineTwoFeatures(pTmpData, numWeakLearners, nearestNeighborIndices[t]);
            }
         }	    
      }
   }

   // if combined feature
   if (_selectedColumn >= pTmpData->getNumColumnsOriginal()) 
   {
      // for debugging:
      cout << "Combined feature" << endl;
      cout << "threshold = " << _threshold << endl;
      for (int i = 0;i < 10;i++)
         cout << "v[" << i << "] = " << _v[i] << endl;
      // end for debugging

      // remember to the parents, will be used in phi for test vectors
      _pParentWeakLearner1 = pTmpData->getParent1WeakLearner(_selectedColumn);
      _pParentWeakLearner2 = pTmpData->getParent2WeakLearner(_selectedColumn);

   }
   
   return 0; // broken
}

// -----------------------------------------------------------------------

void HierarchicalStumpLearner::save(ofstream& outputStream, int numTabs)
{
   // Calling the super-class method
   SingleStumpLearner::save(outputStream, numTabs);
}

// -----------------------------------------------------------------------

void HierarchicalStumpLearner::load(nor_utils::StreamTokenizer& st)
{
   // Calling the super-class method
   SingleStumpLearner::load(st);
}

// -----------------------------------------------------------------------

void HierarchicalStumpLearner::getStateData( vector<float>& data, const string& reason, InputData* pData )
{
   // Calling the super-class method
   SingleStumpLearner::getStateData(data,reason,pData);
}

// ------------------------------------------------------------------------------

void HierarchicalStumpLearner::combineTwoFeatures(InputData* pData, int idx1, int idx2)
{
   cout << "Combining features " << idx1 << " and " << idx2 << endl;

   ExtendableData* pTmpData = static_cast<ExtendableData*>(pData);
   const int numExamples = pTmpData->getNumExamples();

   const HierarchicalStumpLearner* pWeakLearner1 = pTmpData->getWeakLearner(idx1);
   const HierarchicalStumpLearner* pWeakLearner2 = pTmpData->getWeakLearner(idx2);

   // The new feature vector
   vector<double> newColumnVector;

   newColumnVector.reserve(numExamples);

   for (int i = 0; i < numExamples; ++i) 
   {
      double value1 = pWeakLearner1->phi(pTmpData, i);
      double value2 = pWeakLearner2->phi(pTmpData, i);
      newColumnVector.push_back(combineTwoValues(value1, value2));
   }
   // add the new feature as a new column, with the indices of 
   // it's two parents
   pTmpData->addColumn(newColumnVector, idx1, idx2);
}

// ------------------------------------------------------------------------------

double HierarchicalStumpLearner::mutualInformationTerm(int n1, int n10, int n01, int n) const
{
   if (n1 == 0)
      return 0.0;
   double p1 = static_cast<double>(n1)/n;
   double p10 = static_cast<double>(n10)/n;
   double p01 = static_cast<double>(n01)/n;
   return p1 * log(p1 / ((p1 + p10) * (p1 + p01))) / LOG2;
}

// ------------------------------------------------------------------------------

double HierarchicalStumpLearner::combineTwoValues(double value1, double value2) const
{
   if (value1 == value2)
      return 0.0;
   else
      return static_cast<double>(value1);
}

// ------------------------------------------------------------------------------

// overridden to call phi below
float HierarchicalStumpLearner::classify(InputData* pData, int idx, int classIdx)
{
   return _v[classIdx] * phi(pData,idx);
}

// ------------------------------------------------------------------------------

float HierarchicalStumpLearner::phi(InputData* pData,int pointIdx) const
{
   ExtendableData* pTmpData = static_cast<ExtendableData*>(pData);

   // For training points use the columns saved:
   //     getNumColumns() is the number of al columns including combined ones
   // For test points use the original columns if the selected column is 
   // original, otherwise call the parents' phi's and do the same combination 
   // as when we added the column in run(). 
   // In other words: we calculate pData->getValue(pointIdx,_selectedColumn)
   // for the new (combined, non-existant) _selectedColumn's. 
   if (_selectedColumn < pTmpData->getNumAttributes())
      return SingleStumpLearner::phi(pData,pointIdx);
   else 
   {
      double value1 = _pParentWeakLearner1->phi(pTmpData,pointIdx);
      double value2 = _pParentWeakLearner2->phi(pTmpData,pointIdx);

      // this is the old phi
      return SingleStumpLearner::phi(combineTwoValues(value1,value2),0); 
   }
}

   // -----------------------------------------------------------------------

} // end of namespace MultiBoost
