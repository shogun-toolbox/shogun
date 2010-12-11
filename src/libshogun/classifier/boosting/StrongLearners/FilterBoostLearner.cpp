#include <ctime> // for time
#include <cmath> // for exp
#include <fstream> // for ofstream of the step-by-step data
#include <limits>
#include <iomanip> // setprecision

#include "classifier/boosting/Utils/Utils.h" // for addAndCheckExtension
#include "classifier/boosting/Defaults.h" // for defaultLearner
#include "classifier/boosting/IO/OutputInfo.h"
#include "classifier/boosting/Others/Rates.h"
#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/IO/Serialization.h" // to save the found strong hypothesis

#include "classifier/boosting/WeakLearners/BaseLearner.h"
#include "classifier/boosting/StrongLearners/FilterBoostLearner.h"

#include "classifier/boosting/Classifiers/FilterBoostClassifier.h"

namespace MultiBoost {

	// -----------------------------------------------------------------------------------

	void FilterBoostLearner::getArgs(const nor_utils::Args& args)
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

	void FilterBoostLearner::run(const nor_utils::Args& args)
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

		const int numClasses = pTrainingData->getNumClasses();
		const int numExamples = pTrainingData->getNumExamples();
		
		//initialize the margins variable
		_margins.resize( numExamples );
		for( int i=0; i<numExamples; i++ )
		{
			_margins[i].resize( numClasses );
			fill( _margins[i].begin(), _margins[i].end(), 0.0 );
		}


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

			updateMargins( pTrainingData, pConstantWeakHypothesis );

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

			filter( pTrainingData, (int)(_Cn * log(t+2.0)) );
			if ( pTrainingData->getNumExamples() < 2 ) 
			{
				filter( pTrainingData, (int)(_Cn * log(t+2.0)), false );
			}
			
			if (_verbose > 1)
			{
				cout << "--> Size of training data = " << pTrainingData->getNumExamples() << endl;
			}

			BaseLearner* pWeakHypothesis = pWeakHypothesisSource->create();
			pWeakHypothesis->initLearningOptions(args);
			//pTrainingData->clearIndexSet();
			pWeakHypothesis->setTrainingData(pTrainingData);
			float energy = pWeakHypothesis->run();

			BaseLearner* pConstantWeakHypothesis;
			if (_withConstantLearner) // check constant learner if user wants it
			{
				pConstantWeakHypothesis = pConstantWeakHypothesisSource->create() ;
				pConstantWeakHypothesis->initLearningOptions(args);
				pConstantWeakHypothesis->setTrainingData(pTrainingData);
				float constantEnergy = pConstantWeakHypothesis->run();
			}

			//estimate edge
			filter( pTrainingData, (int)(_Cn * log(t+2.0)), false );
			float edge = pWeakHypothesis->getEdge() / 2.0;

			if (_withConstantLearner) // check constant learner if user wants it
			{
				float constantEdge = pConstantWeakHypothesis->getEdge() / 2.0;
				if ( constantEdge > edge )
				{
					delete pWeakHypothesis;
					pWeakHypothesis = pConstantWeakHypothesis;
					edge = constantEdge;
				} else {
					delete pConstantWeakHypothesis;
				}
			}

			// calculate alpha
			float alpha = 0.0;
			alpha = 0.5 * log( ( 0.5 + edge ) / ( 0.5 - edge ) );
			pWeakHypothesis->setAlpha( alpha );

			if (_verbose > 1)
				cout << "Weak learner: " << pWeakHypothesis->getName()<< endl;
			// Output the step-by-step information
			pTrainingData->clearIndexSet();
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

			// update the margins
			updateMargins( pTrainingData, pWeakHypothesis );

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

	void FilterBoostLearner::classify(const nor_utils::Args& args)
	{
		FilterBoostClassifier classifier(args, _verbose);

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

	void FilterBoostLearner::doConfusionMatrix(const nor_utils::Args& args)
	{
		FilterBoostClassifier classifier(args, _verbose);

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

	void FilterBoostLearner::doLikelihoods(const nor_utils::Args& args)
	{
		FilterBoostClassifier classifier(args, _verbose);

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

	void FilterBoostLearner::doPosteriors(const nor_utils::Args& args)
	{
		FilterBoostClassifier classifier(args, _verbose);

		// -posteriors <dataFile> <shypFile> <outFileName>
		string testFileName = args.getValue<string>("posteriors", 0);
		string shypFileName = args.getValue<string>("posteriors", 1);
		string outFileName = args.getValue<string>("posteriors", 2);
		int numIterations = args.getValue<int>("posteriors", 3);

		classifier.savePosteriors(testFileName, shypFileName, outFileName, numIterations);
	}

	// -------------------------------------------------------------------------

	void FilterBoostLearner::doCalibratedPosteriors(const nor_utils::Args& args)
	{
		FilterBoostClassifier classifier(args, _verbose);

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

	float FilterBoostLearner::updateWeights(InputData* pData, BaseLearner* pWeakHypothesis)
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
				//lIt->weight = w * exp( -alpha * _hy[i][lIt->idx] ) / Z;
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

	int FilterBoostLearner::resumeWeakLearners(InputData* pTrainingData)
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

	void FilterBoostLearner::resumeProcess(Serialization& ss, 
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
			updateMargins(pTrainingData, pWeakHypothesis);

			//updateFxs( pTrainingData, pWeakHypothesis );
		
			if (_verbose > 1 && (t + 1) % step == 0)
			{
				float progress = static_cast<float>(t) / static_cast<float>(numIters) * 100.0;                             
				cout << "." << setprecision(2) << progress << "%." << flush;
			}

			// If gamma <= theta there is something really wrong.
			/*
			if (gamma <= _theta)
			{
				cerr << "ERROR!" <<  setprecision(4) << endl
					<< "At iteration <" << t << ">, edge smaller than the edge offset (theta). Something must be wrong!" << endl
					<< "[Edge: " << gamma << " < Offset: " << _theta << "]" << endl
					<< "Is the data file the same one used during the original training?" << endl;
				//          exit(1);
			}
			*/
		}  // loop on iterations

		if (_verbose > 0)
			cout << "Done!" << endl;

	}

	// -------------------------------------------------------------------------

	void FilterBoostLearner::printOutputInfo(OutputInfo* pOutInfo, int t, 
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

	void FilterBoostLearner::doROC(const nor_utils::Args& args)
	{
		FilterBoostClassifier classifier(args, _verbose);

		// -posteriors <dataFile> <shypFile> <outFileName>
		string testFileName = args.getValue<string>("roc", 0);
		string shypFileName = args.getValue<string>("roc", 1);
		string outFileName = args.getValue<string>("roc", 2);
		int numIterations = args.getValue<int>("roc", 3);

		classifier.saveROC(testFileName, shypFileName, outFileName, numIterations);
	}

	// -------------------------------------------------------------------------

	void FilterBoostLearner::filter( InputData* pData, int size, bool rejection )
	{
		pData->clearIndexSet();
		const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();		

		set<int> indexSet;
		//random permutation
		vector< pair<int,int> > tmpRandomArr( numExamples );
		for( int i=0; i < numExamples; i++ ) 
		{
			tmpRandomArr[i].first = rand();
			tmpRandomArr[i].second = i;
		}

		sort( tmpRandomArr.begin(), tmpRandomArr.end(), nor_utils::comparePair<1, int, int, less<int> >() );
		
		vector< int > randPerm( numExamples );
		for( int i=0; i<numExamples; i++ )
		{
			randPerm[i] = tmpRandomArr[i].second;
		}
		//end: random permutation

		int iter = 0;
		int maxIter = 5 * size;
		int wholeIter = 0;

		indexSet.clear();
		while (1)
		{
			if ( size<=indexSet.size() ) break;
			if ( wholeIter > 5 ) rejection = false;
			if ( numExamples <= iter ) {
				iter = 0;
				wholeIter++;
			}

			if ( rejection )
			{				
				const vector<Label>& labels = pData->getLabels( randPerm[iter] );
				vector<Label>::const_iterator lIt;

				float scalar = 0.0;
				//float scalar = numeric_limits<float>::max();
				for ( lIt = labels.begin(); lIt != labels.end(); ++lIt ) 
				{
					//if ( scalar > _margins[ randPerm[iter] ][lIt->idx] ) scalar = _margins[ randPerm[iter] ][lIt->idx];
					//if ( _margins[ randPerm[iter] ][lIt->idx] < 0.0 ) scalar += _margins[ randPerm[iter] ][lIt->idx];
					scalar += _margins[ randPerm[iter] ][lIt->idx];
				}
				
				scalar = scalar / (float) numClasses;

				float randNum = (float)rand() / RAND_MAX;
				float qValue = 1 / ( 1 + exp( scalar ) );

				if ( randNum < qValue ) indexSet.insert( randPerm[iter] );
			}
			else
			{
				indexSet.insert( randPerm[iter] );
			}
			iter++;
		}


		// normalize the weights of the labels
		set<int>::iterator sIt;
		float sum = 0.0;
		// for each example are in use
		for ( sIt = indexSet.begin(); sIt != indexSet.end(); sIt++ )
		{
			vector<Label>& labels = pData->getLabels(*sIt);
			vector<Label>::iterator lIt;

			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				lIt->weight = 1 /( 1+exp( _margins[ *sIt ][lIt->idx] ) );
				sum += lIt->weight;
			}
		}

		for ( sIt = indexSet.begin(); sIt != indexSet.end(); sIt++ )
		{
			vector<Label>& labels = pData->getLabels(*sIt);
			vector<Label>::iterator lIt;

			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				lIt->weight /= sum;
			}
		}

		pData->loadIndexSet( indexSet );
		/*
		sum = 0.0;
		for ( int i=0; i < pData->getNumExamples(); i++ )
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;

			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				sum += lIt->weight;
			}
		}
		cout << "The size of the dataset: " << pData->getNumExamples() << endl;
		cout << "Sum: " << sum << endl;
		*/
	}

	void FilterBoostLearner::updateMargins( InputData* pData, BaseLearner* pWeakHypothesis )
	{
		pData->clearIndexSet();
		const int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();

		// for each example
		for (int i = 0; i < numExamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;

			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				float hy =  pWeakHypothesis->classify(pData, i, lIt->idx) * // h_l(x_i)
					lIt->y; // y

				// compute the margin
				_margins[i][lIt->idx] += pWeakHypothesis->getAlpha() * hy;
			}
		}


	}
} // end of namespace MultiBoost

