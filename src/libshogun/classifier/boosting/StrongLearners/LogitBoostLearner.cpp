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


#include <ctime> // for time
#include <cmath> // for exp
#include <fstream> // for ofstream of the step-by-step data
#include <limits>
#include <iomanip> // setprecision

#include "classifier/boosting/Utils/Utils.h" // for addAndCheckExtension
#include "classifier/boosting/Defaults.h" // for defaultLearner
#include "classifier/boosting/IO/OutputInfo.h"
#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/IO/Serialization.h" // to save the found strong hypothesis

#include "classifier/boosting/WeakLearners/BaseLearner.h"
#include "classifier/boosting/StrongLearners/LogitBoostLearner.h"

#include "classifier/boosting/Classifiers/LogitBoostClassifier.h"

namespace shogun {

	// -----------------------------------------------------------------------------------

	void LogitBoostLearner::getArgs(const nor_utils::Args& args)
	{
		if ( args.hasArgument("verbose") )
			args.getValue("verbose", 0, _verbose);

		// The file with the step-by-step information
		if ( args.hasArgument("outputinfo") )
			args.getValue("outputinfo", 0, _outputInfoFile);

		///////////////////////////////////////////////////
		// get the output strong hypothesis file name, if given
		if ( args.hasArgument("shypname") )
			args.getValue("shypname", 0, _shypFileName);
		else
			_shypFileName = string(SHYP_NAME);

		_shypFileName = nor_utils::addAndCheckExtension(_shypFileName, SHYP_EXTENSION);

		///////////////////////////////////////////////////
		// get the output strong hypothesis file name, if given
		if ( args.hasArgument("shypcomp") )
			args.getValue("shypcomp", 0, _isShypCompressed );
		else
			_isShypCompressed = false;


		///////////////////////////////////////////////////
		// Set time limit
		if ( args.hasArgument("timelimit") )
		{
			args.getValue("timelimit", 0, _maxTime);   
			if (_verbose > 1)    
				cout << "--> Overall Time Limit: " << _maxTime << " minutes" << endl;
		}

		// Set the value of theta
		if ( args.hasArgument("edgeoffset") )
			args.getValue("edgeoffset", 0, _theta);  

		// Set the filename of the strong hypothesis file in the case resume is
		// called
		if ( args.hasArgument("resume") )
			args.getValue("resume", 0, _resumeShypFileName);

		// get the name of the learner
		_baseLearnerName = defaultLearner;
		if ( args.hasArgument("learnertype") )
			args.getValue("learnertype", 0, _baseLearnerName);

		// -train <dataFile> <nInterations>
		if ( args.hasArgument("train") )
		{
			args.getValue("train", 0, _trainFileName);
			args.getValue("train", 1, _numIterations);
		}
		// -traintest <trainingDataFile> <testDataFile> <nInterations>
		else if ( args.hasArgument("traintest") ) 
		{
			args.getValue("traintest", 0, _trainFileName);
			args.getValue("traintest", 1, _testFileName);
			args.getValue("traintest", 2, _numIterations);
		}

		// --constant: check constant learner in each iteration
		if ( args.hasArgument("constant") )
			_withConstantLearner = true;

	}

	// -----------------------------------------------------------------------------------

	void LogitBoostLearner::run(const nor_utils::Args& args)
	{
		// load the arguments
		this->getArgs(args);

		time_t startTime, currentTime;
		time(&startTime);

		// get the registered weak learner (type from name)
		BaseLearner* pWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner(_baseLearnerName);
		// initialize learning options; normally it's done in the strong loop
		// also, here we do it for Product learners, so input data can be created
		pWeakHypothesisSource->initLearningOptions(args);

		BaseLearner* pConstantWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner("ConstantLearner");

		// get the training input data, and load it

		InputData* pTrainingData = pWeakHypothesisSource->createInputData();
		pTrainingData->initOptions(args);
		pTrainingData->load(_trainFileName, IT_TRAIN, _verbose);

		// get the testing input data, and load it
		InputData* pTestData = NULL;
		if ( !_testFileName.empty() )
		{
			pTestData = pWeakHypothesisSource->createInputData();
			pTestData->initOptions(args);
			pTestData->load(_testFileName, IT_TEST, _verbose);
		}

		// The output information object
		OutputInfo* pOutInfo = NULL;


		if ( !_outputInfoFile.empty() ) 
		{
			// Baseline: constant classifier - goes into 0th iteration

			BaseLearner* pConstantWeakHypothesis = pConstantWeakHypothesisSource->create() ;
			pConstantWeakHypothesis->initLearningOptions(args);
			pConstantWeakHypothesis->setTrainingData(pTrainingData);
			float constantEnergy = pConstantWeakHypothesis->run();

			pOutInfo = new OutputInfo(_outputInfoFile);
			pOutInfo->initialize(pTrainingData);

			if (pTestData)
				pOutInfo->initialize(pTestData);
			pOutInfo->outputHeader();

			pOutInfo->outputIteration(-1);
			pOutInfo->outputError(pTrainingData, pConstantWeakHypothesis);

			if (pTestData)
				pOutInfo->outputError(pTestData, pConstantWeakHypothesis);
			/*
			pOutInfo->outputMargins(pTrainingData, pConstantWeakHypothesis);
			pOutInfo->outputEdge(pTrainingData, pConstantWeakHypothesis);

			if (pTestData)
				pOutInfo->outputMargins(pTestData, pConstantWeakHypothesis);

			pOutInfo->outputMAE(pTrainingData);

			if (pTestData)
				pOutInfo->outputMAE(pTestData);
			*/
			pOutInfo->outputCurrentTime();

			pOutInfo->endLine();
			pOutInfo->initialize(pTrainingData);

			if (pTestData)
				pOutInfo->initialize(pTestData);
		}
		// reload the previously found weak learners if -resume is set. 
		// otherwise just return 0
		int startingIteration = resumeWeakLearners(pTrainingData);


		Serialization ss(_shypFileName, _isShypCompressed );
		ss.writeHeader(_baseLearnerName); // this must go after resumeProcess has been called

		// perform the resuming if necessary. If not it will just return
		resumeProcess(ss, pTrainingData, pTestData, pOutInfo);

		if (_verbose == 1)
			cout << "Learning in progress..." << endl;

		///////////////////////////////////////////////////////////////////////
		// Starting the AdaBoost main loop
		///////////////////////////////////////////////////////////////////////
		for (int t = startingIteration; t < _numIterations; ++t)
		{
			if (_verbose > 1)
				cout << "------- WORKING ON ITERATION " << (t+1) << " -------" << endl;

			BaseLearner* pWeakHypothesis = pWeakHypothesisSource->create();
			pWeakHypothesis->initLearningOptions(args);
			//pTrainingData->clearIndexSet();
			pWeakHypothesis->setTrainingData(pTrainingData);
			float energy = pWeakHypothesis->run();

			if (_withConstantLearner) // check constant learner if user wants it
			{
				BaseLearner* pConstantWeakHypothesis = pConstantWeakHypothesisSource->create() ;
				pConstantWeakHypothesis->initLearningOptions(args);
				pConstantWeakHypothesis->setTrainingData(pTrainingData);
				float constantEnergy = pConstantWeakHypothesis->run();

				if (constantEnergy <= energy)
					pWeakHypothesis = pConstantWeakHypothesis;
			}

			if (_verbose > 1)
				cout << "Weak learner: " << pWeakHypothesis->getName()<< endl;
			// Output the step-by-step information
			printOutputInfo(pOutInfo, t, pTrainingData, pTestData, pWeakHypothesis);

			// Updates the weights and returns the edge
			float gamma = updateWeights(pTrainingData, pWeakHypothesis);

			if (_verbose > 1)
			{
				cout << setprecision(5)
					<< "--> Alpha = " << pWeakHypothesis->getAlpha() << endl
					<< "--> Edge  = " << gamma << endl
					<< "--> Energy  = " << energy << endl
					//            << "--> ConstantEnergy  = " << constantEnergy << endl
					//            << "--> difference  = " << (energy - constantEnergy) << endl
					;
			}

			// If gamma <= theta the algorithm must stop.
			// If theta == 0 and gamma is 0, it means that the weak learner is no better than chance
			// and no further training is possible.
			if (gamma <= _theta)
			{
				if (_verbose > 0)
				{
					cout << "Can't train any further: edge = " << gamma 
						<< " (with and edge offset (theta)=" << _theta << ")" << endl;
				}

				//          delete pWeakHypothesis;
				//          break; 
			}

			// append the current weak learner to strong hypothesis file,
			// that is, serialize it.
			ss.appendHypothesis(t, pWeakHypothesis);

			// Add it to the internal list of weak hypotheses
			_foundHypotheses.push_back(pWeakHypothesis); 

			// check if the time limit has been reached
			if (_maxTime > 0)
			{
				time( &currentTime );
				float diff = difftime(currentTime, startTime); // difftime is in seconds
				diff /= 60; // = minutes

				if (diff > _maxTime)
				{
					if (_verbose > 0)
						cout << "Time limit of " << _maxTime 
						<< " minutes has been reached!" << endl;
					break;     
				}
			} // check for maxtime
			delete pWeakHypothesis;
		}  // loop on iterations
		/////////////////////////////////////////////////////////

		// write the footer of the strong hypothesis file
		ss.writeFooter();

		// Free the two input data objects
		if (pTrainingData)
			delete pTrainingData;
		if (pTestData)
			delete pTestData;

		if (pOutInfo)
			delete pOutInfo;

		if (_verbose > 0)
			cout << "Learning completed." << endl;
	}

	// -------------------------------------------------------------------------

	void LogitBoostLearner::classify(const nor_utils::Args& args)
	{
		LogitBoostClassifier classifier(args, _verbose);

		// -test <dataFile> <shypFile>
		string testFileName = args.getValue<string>("test", 0);
		string shypFileName = args.getValue<string>("test", 1);
		int numIterations = args.getValue<int>("test", 2);

		string outResFileName;
		if ( args.getNumValues("test") > 3 )
			args.getValue("test", 3, outResFileName);

		classifier.run(testFileName, shypFileName, numIterations, outResFileName);
	}

	// -------------------------------------------------------------------------

	void LogitBoostLearner::doConfusionMatrix(const nor_utils::Args& args)
	{
		LogitBoostClassifier classifier(args, _verbose);

		// -cmatrix <dataFile> <shypFile>
		if ( args.hasArgument("cmatrix") )
		{
			string testFileName = args.getValue<string>("cmatrix", 0);
			string shypFileName = args.getValue<string>("cmatrix", 1);

			classifier.printConfusionMatrix(testFileName, shypFileName);
		}
		// -cmatrixfile <dataFile> <shypFile> <outFile>
		else if ( args.hasArgument("cmatrixfile") )
		{
			string testFileName = args.getValue<string>("cmatrix", 0);
			string shypFileName = args.getValue<string>("cmatrix", 1);
			string outResFileName = args.getValue<string>("cmatrix", 2);

			classifier.saveConfusionMatrix(testFileName, shypFileName, outResFileName);
		}
	}

	// -------------------------------------------------------------------------

	void LogitBoostLearner::doLikelihoods(const nor_utils::Args& args)
	{
		LogitBoostClassifier classifier(args, _verbose);

		// -posteriors <dataFile> <shypFile> <outFileName>
		string testFileName = args.getValue<string>("likelihood", 0);
		string shypFileName = args.getValue<string>("likelihood", 1);
		string outFileName = args.getValue<string>("likelihood", 2);
		int numIterations = args.getValue<int>("likelihood", 3);

		classifier.saveLikelihoods(testFileName, shypFileName, outFileName, numIterations);
	}

	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------


	// -------------------------------------------------------------------------

	void LogitBoostLearner::doPosteriors(const nor_utils::Args& args)
	{
		LogitBoostClassifier classifier(args, _verbose);

		// -posteriors <dataFile> <shypFile> <outFileName>
		string testFileName = args.getValue<string>("posteriors", 0);
		string shypFileName = args.getValue<string>("posteriors", 1);
		string outFileName = args.getValue<string>("posteriors", 2);
		int numIterations = args.getValue<int>("posteriors", 3);

		classifier.savePosteriors(testFileName, shypFileName, outFileName, numIterations);
	}

	// -------------------------------------------------------------------------

	void LogitBoostLearner::doCalibratedPosteriors(const nor_utils::Args& args)
	{
		LogitBoostClassifier classifier(args, _verbose);

		// -posteriors <dataFile> <shypFile> <outFileName>
		string testFileName = args.getValue<string>("cposteriors", 0);
		string shypFileName = args.getValue<string>("cposteriors", 1);
		string outFileName = args.getValue<string>("cposteriors", 2);
		int numIterations = args.getValue<int>("cposteriors", 3);

		classifier.saveCalibratedPosteriors(testFileName, shypFileName, outFileName, numIterations);
	}


	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------

	float LogitBoostLearner::updateWeights(InputData* pData, BaseLearner* pWeakHypothesis)
	{
		const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();

		const float alpha = pWeakHypothesis->getAlpha();

		float Z = 0; // The normalization factor

		_hy.resize(numExamples);
		for ( int i = 0; i < numExamples; ++i)
			_hy[i].resize(numClasses);
		// recompute weights
		// computing the normalization factor Z

		// for each example
		for (int i = 0; i < numExamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;

			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				_hy[i][lIt->idx] = pWeakHypothesis->classify(pData, i, lIt->idx) * // h_l(x_i)
					lIt->y;
				Z += lIt->weight * // w
					exp( 
					-alpha * _hy[i][lIt->idx] // -alpha * h_l(x_i) * y_i
				);
				// important!
				// _hy[i] must be a vector with different sizes, depending on the
				// example!
				// so it will become:
				// _hy[i][l] 
				// where l is NOT the index of the label (lIt->idx), but the index in the 
				// label vector of the example
			}
		}

		float gamma = 0;

		// Now do the actual re-weight
		// (and compute the edge at the same time)
		// for each example
		for (int i = 0; i < numExamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;

			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				float w = lIt->weight;
				gamma += w * _hy[i][lIt->idx];

				// The new weight is  w * exp( -alpha * h(x_i) * y_i ) / Z
				lIt->weight = w * exp( -alpha * _hy[i][lIt->idx] ) / Z;
			}
		}


		//for (int i = 0; i < numExamples; ++i)
		//{
		//   for (int l = 0; l < numClasses; ++l)
		//   {
		//      _hy[i][l] = pWeakHypothesis->classify(pData, i, l) * // h_l(x_i)
		//                  pData->getLabel(i, l); // y_i

		//      Z += pData->getWeight(i, l) * // w
		//           exp( 
		//             -alpha * _hy[i][l] // -alpha * h_l(x_i) * y_i
		//           );
		//   } // numClasses
		//} // numExamples

		// The edge. It measures the
		// accuracy of the current weak hypothesis relative to random guessing

		//// Now do the actual re-weight
		//// (and compute the edge at the same time)
		//for (int i = 0; i < numExamples; ++i)
		//{
		//   for (int l = 0; l < numClasses; ++l)
		//   {  
		//      float w = pData->getWeight(i, l);

		//      gamma += w * _hy[i][l];

		//      // The new weight is  w * exp( -alpha * h(x_i) * y_i ) / Z
		//      pData->setWeight( i, l, 
		//                        w * exp( -alpha * _hy[i][l] ) / Z );
		//   } // numClasses
		//} // numExamples

		return gamma;
	}

	// -------------------------------------------------------------------------

	int LogitBoostLearner::resumeWeakLearners(InputData* pTrainingData)
	{
		if (_resumeShypFileName.empty())
			return 0;

		if (_verbose > 0)
			cout << "Reloading strong hypothesis file <" << _resumeShypFileName << ">.." << flush;

		// The class that loads the weak hypotheses
		UnSerialization us;

		// loads them
		us.loadHypotheses(_resumeShypFileName, _foundHypotheses, pTrainingData, _verbose);

		if (_verbose > 0)
			cout << "Done!" << endl;

		// return the number of iterations found
		return static_cast<int>( _foundHypotheses.size() );
	}

	// -------------------------------------------------------------------------

	void LogitBoostLearner::resumeProcess(Serialization& ss, 
		InputData* pTrainingData, InputData* pTestData, 
		OutputInfo* pOutInfo)
	{

		if (_resumeShypFileName.empty())
			return;

		if (_verbose > 0)
			cout << "Resuming up to iteration " << _foundHypotheses.size() - 1 << ": 0%." << flush;

		vector<BaseLearner*>::iterator it;
		int t;

		// rebuild the new strong hypothesis file
		for (it = _foundHypotheses.begin(), t = 0; it != _foundHypotheses.end(); ++it, ++t)
		{
			BaseLearner* pWeakHypothesis = *it;

			// append the current weak learner to strong hypothesis file,
			ss.appendHypothesis(t, pWeakHypothesis);
		}

		const int numIters = static_cast<int>(_foundHypotheses.size());
		const int step = numIters < 5 ? 1 : numIters / 5;

		// simulate the AdaBoost algorithm for the weak learners already found
		for (it = _foundHypotheses.begin(), t = 0; it != _foundHypotheses.end(); ++it, ++t)
		{
			BaseLearner* pWeakHypothesis = *it;

			// Output the step-by-step information
			printOutputInfo(pOutInfo, t, pTrainingData, pTestData, pWeakHypothesis);

			// Updates the weights and returns the edge
			float gamma = updateWeights(pTrainingData, pWeakHypothesis);

			if (_verbose > 1 && (t + 1) % step == 0)
			{
				float progress = static_cast<float>(t) / static_cast<float>(numIters) * 100.0;                             
				cout << "." << setprecision(2) << progress << "%." << flush;
			}

			// If gamma <= theta there is something really wrong.
			if (gamma <= _theta)
			{
				cerr << "ERROR!" <<  setprecision(4) << endl
					<< "At iteration <" << t << ">, edge smaller than the edge offset (theta). Something must be wrong!" << endl
					<< "[Edge: " << gamma << " < Offset: " << _theta << "]" << endl
					<< "Is the data file the same one used during the original training?" << endl;
				//          exit(1);
			}

		}  // loop on iterations

		if (_verbose > 0)
			cout << "Done!" << endl;

	}

	// -------------------------------------------------------------------------

	void LogitBoostLearner::printOutputInfo(OutputInfo* pOutInfo, int t, 
		InputData* pTrainingData, InputData* pTestData, 
		BaseLearner* pWeakHypothesis)
	{

		pOutInfo->outputIteration(t);
		pOutInfo->outputError(pTrainingData, pWeakHypothesis);
		if (pTestData)
			pOutInfo->outputError(pTestData, pWeakHypothesis);
		/*
		pOutInfo->outputMargins(pTrainingData, pWeakHypothesis);
		pOutInfo->outputEdge(pTrainingData, pWeakHypothesis);
		if (pTestData)
			pOutInfo->outputMargins(pTestData, pWeakHypothesis);
		pOutInfo->outputMAE(pTrainingData);      
		if (pTestData)
			pOutInfo->outputMAE(pTestData);  
		*/
		pOutInfo->outputCurrentTime();
		pOutInfo->endLine();
	}

	// -------------------------------------------------------------------------

} // end of namespace shogun
