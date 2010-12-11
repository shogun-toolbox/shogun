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
* \file ADTreeData.h Extension of SortedData to deal with ADTrees.
*/

#ifndef __ADTREE_DATA_H
#define __ADTREE_DATA_H

#include "classifier/boosting/IO/SortedData.h"

using namespace std;

namespace MultiBoost {

class BaseLearner;
class ADTreePredictionNode;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

/**
* Input data class specifically defined for ADTree.
* This class has an only purpose: hold the list of the prediction nodes
* found during the training process, so that the ADTree weak learner can use
* it instead of requesting it from the strong learner.
*/
class ADTreeData : public SortedData
{
public:

   /**
   * Add a prediction node (its pointer) to the list.
   * \param predNode the prediction node to be added.
   */
   void addPredictionNode(ADTreePredictionNode* predNode) { _predictionNodesFound.push_back(predNode); }

   /**
   * Get the list of the prediction nodes found so far.
   * \return A const vector of the prediction nodes found so far.
   */
   const vector<ADTreePredictionNode*>& getPredictionNodes() { return _predictionNodesFound; }

protected:

   /**
   * The list of the prediction nodes.
   */
   vector<ADTreePredictionNode*> _predictionNodesFound;

};

} // end of namespace MultiBoost

#endif // __ADTREE_DATA_H
