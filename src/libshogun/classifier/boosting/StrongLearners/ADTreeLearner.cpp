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

#ifdef MBFIXME

#include "classifier/boosting/StrongLearners/ADTreeLearner.h"

#include <cassert>
#include <cmath> // exp

#include "classifier/boosting/Defaults.h" // for SHYP_NAME
#include "classifier/boosting/Utils/Utils.h" // for addAndCheckExtension

#include "classifier/boosting/WeakLearners/ADTree/ADTreeWeakLearner.h"
#include "classifier/boosting/WeakLearners/BaseLearner.h"

#include "classifier/boosting/IO/ADTreeData.h"
#include "classifier/boosting/IO/OutputInfo.h"
#include "classifier/boosting/IO/Serialization.h"

namespace shogun {

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

ADTreeLearner::ADTreeArgs ADTreeLearner::getArgs(const nor_utils::Args& args)
{
   ADTreeArgs retArgs;

   if ( args.hasArgument("verbose") )
      args.getValue( "verbose", 0, retArgs.verbose );

   // The file with the step-by-step information
   if ( args.hasArgument("outputinfo") )
        args.getValue("outputinfo", 0, retArgs.outputInfoFile);

   ///////////////////////////////////////////////////
   // get the output strong hypothesis file name, if given
   if ( args.hasArgument("shypname") )
        args.getValue("shypname", 0, retArgs.shypFileName);
   else
      retArgs.shypFileName = string(SHYP_NAME);

   retArgs.shypFileName = nor_utils::addAndCheckExtension(retArgs.shypFileName, SHYP_EXTENSION);

   ///////////////////////////////////////////////////
   // Set time limit
   if ( args.hasArgument("timelimit") )
   {
      args.getValue("timelimit", 0, retArgs.maxTime);   
      if (retArgs.verbose > 1)    
         cout << "--> Overall Time Limit: " << retArgs.maxTime << " minutes" << endl;
   }
   else
      retArgs.maxTime = -1; // no limits

   // Set the value of theta
   if ( args.hasArgument("edgeoffset") )
      args.getValue("edgeoffset", 0, retArgs.theta);
   else
      retArgs.theta = 0;

   // Set the filename of the strong hypothesis file in the case resume is
   // called
   if ( args.hasArgument("resume") )
      args.getValue("resume", 0, retArgs.resumeShypFileName);

   // get the name of the learner
   retArgs.baseLearnerName = "ADTreeLearner";
   if ( args.hasArgument("learnertype") )
      args.getValue("learnertype", 0, retArgs.baseLearnerName);

   // -train <dataFile> <nInterations>
   if ( args.hasArgument("train") )
   {
      args.getValue("train", 0, retArgs.trainFileName);
      args.getValue("train", 1, retArgs.numIterations);
   }
   // -traintest <trainingDataFile> <testDataFile> <nInterations>
   else if ( args.hasArgument("traintest") ) 
   {
      args.getValue("traintest", 0, retArgs.trainFileName);
      args.getValue("traintest", 1, retArgs.testFileName);
      args.getValue("traintest", 2, retArgs.numIterations);
   }

   return retArgs;
}

// -------------------------------------------------------------------------

void ADTreeLearner::run(const nor_utils::Args& args)
{
   // load the arguments
   ADTreeArgs readedArgs = this->getArgs(args);

   time_t startTime, currentTime;
   time(&startTime);

   // get the registered weak learner (type from name)
   BaseLearner* pWeakHypothesisSource = 
       BaseLearner::RegisteredLearners().getLearner(readedArgs.baseLearnerName);

   // get the training input data, and load it
   // NOTE: Always using ADTree data for training in ADTreeLearner.. :)
   ADTreeData* pTrainingData = static_cast<ADTreeData*>( pWeakHypothesisSource->createInputData() );
   pTrainingData->initOptions(args);
   pTrainingData->load(readedArgs.trainFileName, IT_TRAIN, readedArgs.verbose);

   // get the testing input data, and load it
   InputData* pTestData = NULL;
   if ( !readedArgs.testFileName.empty() )
   {
      pTestData = pWeakHypothesisSource->createInputData();
      pTestData->initOptions(args);
      pTestData->load(readedArgs.testFileName, IT_TEST, readedArgs.verbose);
   }

   // The output information object
   OutputInfo* pOutInfo = NULL;

   if ( !readedArgs.outputInfoFile.empty() )
      pOutInfo = new OutputInfo(readedArgs.outputInfoFile);

   // reload the previously found weak learners if -resume is set. 
   // otherwise just return 0
   int startingIteration = 0; //resumeWeakLearners();

   Serialization ss(readedArgs.shypFileName);
   ss.writeHeader(readedArgs.baseLearnerName); // this must go after resumeProcess has been called

   // perform the resuming if necessary. If not it will just return
   //resumeProcess(ss, pTrainingData, pTestData, pOutInfo);

   if (readedArgs.verbose == 1)
      cout << "Learning in progress..." << endl;

   // Now create the head with the precondition and the values of the given source.
   // Note that the data source must have the weights initialized and not modified.
   ADTreeWeakLearner* pTmp = dynamic_cast<ADTreeWeakLearner*>(pWeakHypothesisSource);
   assert(pTmp != NULL);

   _pTreeHead = pTmp->createHead(pTrainingData);
   _foundRules.push_back(_pTreeHead);
   
   ///////////////////////////////////////////////////////////////////////
   // Starting the AdaBoost main loop
   ///////////////////////////////////////////////////////////////////////
   for (int t = startingIteration; t < readedArgs.numIterations; ++t)
   {
      if (readedArgs.verbose > 1)
         cout << "------- WORKING ON RULE " << t+1 << " -------" << endl;

      BaseLearner* pWeakHypothesis = pWeakHypothesisSource->create();
      pWeakHypothesis->initOptions(args);
      pWeakHypothesis->setTrainingData(pTrainingData);
      pWeakHypothesis->run();

      // Output the step-by-step informations
      if ( pOutInfo )
      {
         pOutInfo->outputIteration(t);
         pOutInfo->outputError(pTrainingData, pWeakHypothesis);
         if (pTestData)
            pOutInfo->outputError(pTestData, pWeakHypothesis);
         pOutInfo->outputMargins(pTrainingData, pWeakHypothesis);
         pOutInfo->outputEdge(pTrainingData, pWeakHypothesis);
         pOutInfo->endLine();
      }

      // Updates the weights and returns the edge
      double gamma = updateWeights(pTrainingData, pWeakHypothesis);

      if (readedArgs.verbose > 1)
      {
         cout << setprecision(5)
            << "--> Edge  = " << gamma << endl;
      }

      // If gamma <= theta the algorithm must stop.
      // If theta == 0 and gamma is 0, it means that the weak learner is no better than chance
      // and no further training is possible.
      if (gamma <= readedArgs.theta)
      {
         ADTreeWeakLearner* pNode = dynamic_cast<ADTreeWeakLearner*>(pWeakHypothesis);
         pNode->makeOrphan();
         
         if (readedArgs.verbose > 0)
         {
            cout << "Can't train any further: edge = " << gamma 
               << " (with and edge offset (theta)=" << readedArgs.theta << ")" << endl;
         }

         delete pWeakHypothesis;
         break; 
      }

      // append the current weak learner to strong hypothesis file,
      // that is, serialize it.
      //ss.appendHypothesis(t, pWeakHypothesis);

      // Add it to the internal list of weak hypotheses
      _foundRules.push_back(pWeakHypothesis);

      // check if the time limit has been reached
      if (readedArgs.maxTime > 0)
      {
         time( &currentTime );
         double diff = difftime(currentTime, startTime); // difftime is in seconds
         diff /= 60; // = minutes

         if (diff > readedArgs.maxTime)
         {
            if (readedArgs.verbose > 0)
               cout << "Time limit of " << readedArgs.maxTime 
               << " minutes has been reached!" << endl;
            break;     
         }
      } // check for maxtime

   }  // loop on iterations
   /////////////////////////////////////////////////////////

   // write the footer of the strong hypothesis file
   ss.writeFooter();

   // DEBUG ONLY
   createGraphvizDot("adtree_graph.dot", _pTreeHead, pTrainingData);

   // Free the two input data objects
   if (pTrainingData)
      delete pTrainingData;
   if (pTestData)
      delete pTestData;

   if (pOutInfo)
      delete pOutInfo;

   if (readedArgs.verbose > 0)
      cout << "Learning completed." << endl;
}

// -------------------------------------------------------------------------

double ADTreeLearner::updateWeights(InputData* pTrainingData, BaseLearner* pWeakHypothesis)
{
   const int numExamples = pTrainingData->getNumExamples();
   const int numClasses = pTrainingData->getNumClasses();

   // The normalization factor
   double Z = 0;
   _ry.resize(numExamples);
   for ( int i = 0; i < numExamples; ++i)
      _ry[i].resize(numClasses);

   //ADTreeWeakLearner* pNode = dynamic_cast<ADTreeWeakLearner*>(pWeakHypothesis);
   //const vector<int>& partIdxs = pNode->getParent()->getPartitionIdxs();
   //vector<int>::const_iterator it;
   //for (int i = 0; i < numExamples; ++i)
   //{
   //   for (int l = 0; l < numClasses; ++l)
   //   {  
   //      _ry[i][l] = pWeakHypothesis->classify(pTrainingData, i, l) * // r_l(x_i) = the rule
   //                  pTrainingData->getBinaryClass(i, l); // y_i

   //      Z += pTrainingData->getWeight(i, l) * exp( -_ry[i][l] );

   //   } // numClasses
   //} // numExamples

   //// The edge. It measures the
   //// accuracy of the current weak hypothesis relative to random guessing
   //double gamma = 0;

   //// Now do the actual re-weight
   //// (and compute the edge at the same time)
   //for (it = partIdxs.begin(); it != partIdxs.end(); ++it)
   //{
   //   for (int l = 0; l < numClasses; ++l)
   //   {  
   //      double w = pTrainingData->getWeight(*it, l);

   //      gamma += w * _ry[*it][l];

   //      // The new weight is  w * exp( -r_l(x_i) * y_i ) / Z
   //      pTrainingData->setWeight( *it, l, w * exp(-_ry[*it][l]) / Z );

   //      //         W += pTrainingData->getWeight(i, l);

   //   } // numClasses
   //} // numExamples

   // Now do the actual re-weight
   // (and compute the edge at the same time)
   for (int i = 0; i < numExamples; ++i)
   {
      for (int l = 0; l < numClasses; ++l)
      {  
         _ry[i][l] = pWeakHypothesis->classify(pTrainingData, i, l) * // r_l(x_i) = the rule
                     pTrainingData->getLabel(i, l); // y_i

         Z += pTrainingData->getWeight(i, l) * exp( -_ry[i][l] );

      } // numClasses
   } // numExamples

   // The edge. It measures the
   // accuracy of the current weak hypothesis relative to random guessing
   double gamma = 0;

   // Now do the actual re-weight
   // (and compute the edge at the same time)
   for (int i = 0; i < numExamples; ++i)
   {
      for (int l = 0; l < numClasses; ++l)
      {  
         double w = pTrainingData->getWeight(i, l);
        
         gamma += w * _ry[i][l];

         // The new weight is  w * exp( -r_l(x_i) * y_i ) / Z
         pTrainingData->setWeight( i, l, w * exp(-_ry[i][l]) / Z );

      } // numClasses
   } // numExamples

   //double W = 0;
   //for (int i = 0; i < numExamples; ++i)
   //{
   //   for (int l = 0; l < numClasses; ++l)
   //   {  
   //      W += pTrainingData->getWeight(i, l);
   //   } // numClasses
   //} // numExamples

   return gamma;
}

// -------------------------------------------------------------------------

void ADTreeLearner::createGraphvizDot(const string& outFileName, BaseLearner* pTreeHead, /*TEMP*/InputData* pData)
{
   ofstream dotFile(outFileName.c_str());

   dotFile << "digraph G {\n";

   dotFile << "\torientation=\"landscape\";\n";
   dotFile << "\tsize=\"10,7.5\";\n";
   dotFile << "\tratio=\"compress\";\n\n";

   recursivePrintDotNode( dotFile, dynamic_cast<ADTreeWeakLearner*>(pTreeHead), pData, true );

   dotFile << "}" << endl;
}

// -------------------------------------------------------------------------

void ADTreeLearner::recursivePrintDotNode(ostream& dotFile, ADTreeWeakLearner* node, /*TEMP*/InputData* pData, bool isHead)
{
   const int numClasses = pData->getNumClasses();
   // first the pre-condition

   const string tab = "\t";

   // name and shape
   if (!isHead)
   {
      dotFile << tab << node->getId() << " [shape=box, label=\"" 
              << node->getConditionString() << "\"];\n";

      // prediction to left
      dotFile << tab << node->getId() << " -> " << "left_" << node->getId()
              << " [label=\"" << node->getGoLeftString() << "\"];\n";

      // prediction to right
      dotFile << tab << node->getId() << " -> " << "right_" << node->getId()
              << " [label=\"" << node->getGoRightString() << "\"];\n";
   }

   // now the connection with the two children

   // left node

   // set up the "name" of the node (with the prediction values)
   vector<double>& leftPreds = node->getLeftPredNode().getPredictionValues();
   assert( static_cast<int>(leftPreds.size()) == numClasses );

   dotFile << tab << "left_" << node->getId() << " [label=\"";
   for (int l = 0; l < numClasses; ++l)
   {
      dotFile << pData->getClassMap().getNameFromIdx(l) << "=" 
              << setprecision(4) << leftPreds[l]               
              << " (" << node->getLeftPredNode().getPartCountPerClass(l, pData) << ")"
              << "\\n";
   }  
   
   dotFile << "(items: " << node->getLeftPredNode().getPartitionSize() << ")\"];\n";

   // right node (not if it is head)
   if (!isHead)
   {
      vector<double>& rightPreds = node->getRightPredNode().getPredictionValues();
      assert( static_cast<int>(rightPreds.size()) == numClasses );

      dotFile << tab << "right_" << node->getId() << " [label=\"";
      for (int l = 0; l < numClasses; ++l)
      {
         dotFile << pData->getClassMap().getNameFromIdx(l) << "=" 
                 << setprecision(4) << rightPreds[l] 
                 << " (" << node->getRightPredNode().getPartCountPerClass(l, pData) << ")"
                 << "\\n";
      }

      dotFile << "(items: " << node->getRightPredNode().getPartitionSize() << ")\"];\n";

   }

   dotFile << "\n";

   // now do recursion!
   vector<ADTreeWeakLearner*>::const_iterator precIt;

   // left recursion
   vector<ADTreeWeakLearner*>& leftChildren = node->getLeftPredNode().getChildren();
   for (precIt = leftChildren.begin(); precIt != leftChildren.end(); ++precIt)
   {
      dotFile << tab << "left_" << node->getId() << " -> " 
              << (*precIt)->getId() << " [style=dashed, label=\"Rule " << (*precIt)->getId() << "\"];\n";
      recursivePrintDotNode(dotFile, *precIt, pData);
   }

   if (!isHead)
   {
      vector<ADTreeWeakLearner*>& rightChildren = node->getRightPredNode().getChildren();
      for (precIt = rightChildren.begin(); precIt != rightChildren.end(); ++precIt)
      {
         dotFile << tab << "right_" << node->getId() << " -> " 
                 << (*precIt)->getId() << " [style=dashed, label=\"Rule " << (*precIt)->getId() << "\"];\n";
         recursivePrintDotNode(dotFile, *precIt, pData);
      }
   }


}

// -------------------------------------------------------------------------

} // end of namespace shogun
#endif
