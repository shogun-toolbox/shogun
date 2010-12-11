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


#include "classifier/boosting/WeakLearners/BaseLearner.h"
#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/Utils/Utils.h"
#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/IO/OutputInfo.h"
#include "classifier/boosting/Classifiers/LogitBoostClassifier.h"
#include "classifier/boosting/Classifiers/ExampleResults.h"

#include "classifier/boosting/WeakLearners/SingleStumpLearner.h" // for saveSingleStumpFeatureData

#include <iomanip> // for setw
#include <cmath> // for setw

namespace shogun {

	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------

	LogitBoostClassifier::LogitBoostClassifier(const nor_utils::Args &args, int verbose)
		: _verbose(verbose), _args(args)
	{
		// The file with the step-by-step information
		if ( args.hasArgument("outputinfo") )
			args.getValue("outputinfo", 0, _outputInfoFile);
	}

	// -------------------------------------------------------------------------

	void LogitBoostClassifier::run(const string& dataFileName, const string& shypFileName, 
		int numIterations, const string& outResFileName, int numRanksEnclosed)
	{
		InputData* pData = loadInputData(dataFileName, shypFileName);

		if (_verbose > 0)
			cout << "Loading strong hypothesis..." << flush;

		// The class that loads the weak hypotheses
		UnSerialization us;

		// Where to put the weak hypotheses
		vector<BaseLearner*> weakHypotheses;

		// loads them
		us.loadHypotheses(shypFileName, weakHypotheses, pData);

		// where the results go
		vector< ExampleResults* > results;

		if (_verbose > 0)
			cout << "Classifying..." << flush;

		// get the results
		computeResults( pData, weakHypotheses, results, numIterations );

		const int numClasses = pData->getNumClasses();

		if (_verbose > 0)
		{
			// well.. if verbose = 0 no results are displayed! :)
			cout << "Done!" << endl;

			vector< vector<float> > rankedError(numRanksEnclosed);

			// Get the per-class error for the numRanksEnclosed-th ranks
			for (int i = 0; i < numRanksEnclosed; ++i)
				getClassError( pData, results, rankedError[i], i );

			// output it
			cout << endl;
			cout << "Error Summary" << endl;
			cout << "=============" << endl;

			for ( int l = 0; l < numClasses; ++l )
			{
				// first rank (winner): rankedError[0]
				cout << "Class '" << pData->getClassMap().getNameFromIdx(l) << "': "
					<< setprecision(4) << rankedError[0][l] * 100 << "%";

				// output the others on its side
				if (numRanksEnclosed > 1 && _verbose > 1)
				{
					cout << " (";
					for (int i = 1; i < numRanksEnclosed; ++i)
						cout << " " << i+1 << ":[" << setprecision(4) << rankedError[i][l] * 100 << "%]";
					cout << " )";
				}

				cout << endl;
			}

			// the overall error
			cout << "\n--> Overall Error: " 
				<< setprecision(4) << getOverallError(pData, results, 0) * 100 << "%";

			// output the others on its side
			if (numRanksEnclosed > 1 && _verbose > 1)
			{
				cout << " (";
				for (int i = 1; i < numRanksEnclosed; ++i)
					cout << " " << i+1 << ":[" << setprecision(4) << getOverallError(pData, results, i) * 100 << "%]";
				cout << " )";
			}

			cout << endl;

		} // verbose


		// If asked output the results
		if ( !outResFileName.empty() )
		{
			const int numExamples = pData->getNumExamples();
			ofstream outRes(outResFileName.c_str());

			string exampleName;

			for (int i = 0; i < numExamples; ++i)
			{
				// output the name if it exists, otherwise the number
				// of the example
				exampleName = pData->getExampleName(i);
				if ( exampleName.empty() )
					outRes << i << '\t';
				else
					outRes << exampleName << '\t';

				// output the predicted class
				outRes << pData->getClassMap().getNameFromIdx( results[i]->getWinner().first ) << endl;
			}

			if (_verbose > 0)
				cout << "\nPredictions written on file <" << outResFileName << ">!" << endl;

		}


		// delete the input data file
		if (pData) 
			delete pData;

		vector<ExampleResults*>::iterator it;
		for (it = results.begin(); it != results.end(); ++it)
			delete (*it);
	}

	// -------------------------------------------------------------------------

	void LogitBoostClassifier::printConfusionMatrix(const string& dataFileName, const string& shypFileName)
	{
		InputData* pData = loadInputData(dataFileName, shypFileName);

		if (_verbose > 0)
			cout << "Loading strong hypothesis..." << flush;

		// The class that loads the weak hypotheses
		UnSerialization us;

		// Where to put the weak hypotheses
		vector<BaseLearner*> weakHypotheses;

		// loads them
		us.loadHypotheses(shypFileName, weakHypotheses, pData);

		// where the results go
		vector< ExampleResults* > results;

		if (_verbose > 0)
			cout << "Classifying..." << flush;

		// get the results
		computeResults( pData, weakHypotheses, results, weakHypotheses.size());

		const int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();

		if (_verbose > 0)
			cout << "Done!" << endl;

		const int colSize = 7;

		if (_verbose > 0)
		{
			cout << "Raw Confusion Matrix:\n";
			cout << setw(colSize) << "Truth       ";

			for (int l = 0; l < numClasses; ++l)
				cout << setw(colSize) << nor_utils::getAlphanumeric(l);

			cout << "\nClassification\n";

			for (int l = 0; l < numClasses; ++l)
			{
				vector<int> winnerCount(numClasses, 0);
				for (int i = 0; i < numExamples; ++i)
				{
					if ( pData->hasPositiveLabel(i, l) )
						++winnerCount[ results[i]->getWinner().first ];
				}

				// class
				cout << setw(colSize) << "           " << nor_utils::getAlphanumeric(l);

				for (int j = 0; j < numClasses; ++j)
					cout << setw(colSize) << winnerCount[j];

				cout << endl;
			}

		}

		cout << "\nMatrix Key:\n";

		// Print the legend
		for (int l = 0; l < numClasses; ++l)
			cout << setw(5) << nor_utils::getAlphanumeric(l) << ": " << 
			pData->getClassMap().getNameFromIdx(l) << "\n";

		// delete the input data file
		if (pData) 
			delete pData;

		vector<ExampleResults*>::iterator it;
		for (it = results.begin(); it != results.end(); ++it)
			delete (*it);
	}

	// -------------------------------------------------------------------------

	void LogitBoostClassifier::saveConfusionMatrix(const string& dataFileName, const string& shypFileName,
		const string& outFileName)
	{
		InputData* pData = loadInputData(dataFileName, shypFileName);

		if (_verbose > 0)
			cout << "Loading strong hypothesis..." << flush;

		// The class that loads the weak hypotheses
		UnSerialization us;

		// Where to put the weak hypotheses
		vector<BaseLearner*> weakHypotheses;

		// loads them
		us.loadHypotheses(shypFileName, weakHypotheses, pData);

		// where the results go
		vector< ExampleResults* > results;

		if (_verbose > 0)
			cout << "Classifying..." << flush;

		// get the results
		computeResults( pData, weakHypotheses, results, weakHypotheses.size() );

		const int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();

		ofstream outFile(outFileName.c_str());

		//////////////////////////////////////////////////////////////////////////

		for (int l = 0; l < numClasses; ++l)
			outFile << '\t' << pData->getClassMap().getNameFromIdx(l);
		outFile << endl;

		for (int l = 0; l < numClasses; ++l)
		{
			vector<int> winnerCount(numClasses, 0);
			for (int i = 0; i < numExamples; ++i)
			{
				if ( pData->hasPositiveLabel(i,l) )
					++winnerCount[ results[i]->getWinner().first ];
			}

			// class name
			outFile << pData->getClassMap().getNameFromIdx(l);

			for (int j = 0; j < numClasses; ++j)
				outFile << '\t' << winnerCount[j];

			outFile << endl;
		}

		//////////////////////////////////////////////////////////////////////////

		if (_verbose > 0)
			cout << "Done!" << endl;

		// delete the input data file
		if (pData) 
			delete pData;

		vector<ExampleResults*>::iterator it;
		for (it = results.begin(); it != results.end(); ++it)
			delete (*it);
	}

	// -------------------------------------------------------------------------

	void LogitBoostClassifier::savePosteriors(const string& dataFileName, const string& shypFileName, 
		const string& outFileName, int numIterations)
	{
		InputData* pData = loadInputData(dataFileName, shypFileName);

		if (_verbose > 0)
			cout << "Loading strong hypothesis..." << flush;

		// The class that loads the weak hypotheses
		UnSerialization us;

		// Where to put the weak hypotheses
		vector<BaseLearner*> weakHypotheses;

		// loads them
		us.loadHypotheses(shypFileName, weakHypotheses, pData);

		// where the results go
		vector< ExampleResults* > results;

		if (_verbose > 0)
			cout << "Classifying..." << flush;

		// get the results
		computeResults( pData, weakHypotheses, results, numIterations );

		const int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();

		ofstream outFile(outFileName.c_str());
		string exampleName;

		if (_verbose > 0)
			cout << "Output posteriors..." << flush;

		for (int i = 0; i < numExamples; ++i)
		{
			// output the name if it exists, otherwise the number
			// of the example
			exampleName = pData->getExampleName(i);
			if ( !exampleName.empty() )
				outFile << exampleName << ',';

			// output the posteriors
			outFile << results[i]->getVotesVector()[0];
			for (int l = 1; l < numClasses; ++l)
				outFile << ',' << results[i]->getVotesVector()[l];
			outFile << '\n';
		}

		if (_verbose > 0)
			cout << "Done!" << endl;

		if (_verbose > 1)
		{
			cout << "\nClass order (You can change it in the header of the data file):" << endl;
			for (int l = 0; l < numClasses; ++l)
				cout << "- " << pData->getClassMap().getNameFromIdx(l) << endl;
		}

		// delete the input data file
		if (pData) 
			delete pData;

		vector<ExampleResults*>::iterator it;
		for (it = results.begin(); it != results.end(); ++it)
			delete (*it);
	}


	// -------------------------------------------------------------------------


	void LogitBoostClassifier::saveCalibratedPosteriors(const string& dataFileName, const string& shypFileName, 
		const string& outFileName, int numIterations)
	{
		InputData* pData = loadInputData(dataFileName, shypFileName);

		if (_verbose > 0)
			cout << "Loading strong hypothesis..." << flush;

		// The class that loads the weak hypotheses
		UnSerialization us;

		// Where to put the weak hypotheses
		vector<BaseLearner*> weakHypotheses;

		// loads them
		us.loadHypotheses(shypFileName, weakHypotheses, pData);

		// where the results go
		vector< ExampleResults* > results;

		if (_verbose > 0)
			cout << "Classifying..." << flush;

		// get the results
		computeResults( pData, weakHypotheses, results, numIterations );

		const int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();

		ofstream outFile(outFileName.c_str());
		string exampleName;

		if (_verbose > 0)
			cout << "Output posteriors..." << flush;

		for (int i = 0; i < numExamples; ++i)
		{
			// output the name if it exists, otherwise the number
			// of the example
			exampleName = pData->getExampleName(i);
			if ( !exampleName.empty() )
				outFile << exampleName << ',';

			// output the posteriors
			outFile << results[i]->getVotesVector()[0];
			for (int l = 1; l < numClasses; ++l)
				outFile << ',' << results[i]->getVotesVector()[l];
			outFile << '\n';
		}

		if (_verbose > 0)
			cout << "Done!" << endl;

		if (_verbose > 1)
		{
			cout << "\nClass order (You can change it in the header of the data file):" << endl;
			for (int l = 0; l < numClasses; ++l)
				cout << "- " << pData->getClassMap().getNameFromIdx(l) << endl;
		}

		// delete the input data file
		if (pData) 
			delete pData;

		vector<ExampleResults*>::iterator it;
		for (it = results.begin(); it != results.end(); ++it)
			delete (*it);
	}



	// -------------------------------------------------------------------------

	void LogitBoostClassifier::saveLikelihoods(const string& dataFileName, const string& shypFileName, 
		const string& outFileName, int numIterations)
	{
		InputData* pData = loadInputData(dataFileName, shypFileName);

		if (_verbose > 0)
			cout << "Loading strong hypothesis..." << flush;

		// The class that loads the weak hypotheses
		UnSerialization us;

		// Where to put the weak hypotheses
		vector<BaseLearner*> weakHypotheses;

		// loads them
		us.loadHypotheses(shypFileName, weakHypotheses, pData);

		// where the results go
		vector< ExampleResults* > results;

		if (_verbose > 0)
			cout << "Classifying..." << flush;

		const int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();


		ofstream outFile(outFileName.c_str());
		string exampleName;

		if (_verbose > 0)
			cout << "Output likelihoods..." << flush;

		// get the results
		/////////////////////////////////////////////////////////////////////
		// computeResults( pData, weakHypotheses, results, numIterations );
		assert( !weakHypotheses.empty() );

		// Initialize the output info
		OutputInfo* pOutInfo = NULL;

		if ( !_outputInfoFile.empty() )
			pOutInfo = new OutputInfo(_outputInfoFile);

		// Creating the results structures. See file Structures.h for the
		// PointResults structure
		results.clear();
		results.reserve(numExamples);
		for (int i = 0; i < numExamples; ++i)
			results.push_back( new ExampleResults(i, numClasses) );

		// sum votes for classes
		vector< float > votesForExamples( numClasses );
		vector< double > expVotesForExamples( numClasses );

		// iterator over all the weak hypotheses
		vector<BaseLearner*>::const_iterator whyIt;
		int t;

		pOutInfo->initialize( pData );

		// for every feature: 1..T
		for (whyIt = weakHypotheses.begin(), t = 0; 
			whyIt != weakHypotheses.end() && t < numIterations; ++whyIt, ++t)
		{
			BaseLearner* currWeakHyp = *whyIt;
			float alpha = currWeakHyp->getAlpha();

			// for every point
			for (int i = 0; i < numExamples; ++i)
			{
				// a reference for clarity and speed
				vector<float>& currVotesVector = results[i]->getVotesVector();

				// for every class
				for (int l = 0; l < numClasses; ++l)
					currVotesVector[l] += alpha * currWeakHyp->classify(pData, i, l);
			}

			// if needed output the step-by-step information
			if ( pOutInfo )
			{
				pOutInfo->outputIteration(t);
				pOutInfo->outputError(pData, currWeakHyp);

				// Margins and edge requires an update of the weight,
				// therefore I keep them out for the moment
				//outInfo.outputMargins(pData, currWeakHyp);
				//outInfo.outputEdge(pData, currWeakHyp);
				pOutInfo->endLine();
			} // for (int i = 0; i < numExamples; ++i)
			// calculate likelihoods from votes

			fill( votesForExamples.begin(), votesForExamples.end(), 0.0 );
			double lLambda = 0.0;
			for (int i = 0; i < numExamples; ++i)
			{
				// a reference for clarity and speed
				vector<float>& currVotesVector = results[i]->getVotesVector();
				double sumExp = 0.0;
				// for every class
				for (int l = 0; l < numClasses; ++l) 
				{				 
					expVotesForExamples[l] =  exp( currVotesVector[l] ) ;
					sumExp += expVotesForExamples[l];
				}			

				if ( sumExp > numeric_limits<double>::epsilon() ) 
				{
					for (int l = 0; l < numClasses; ++l) 
					{
						expVotesForExamples[l] /= sumExp;
					}
				}

				Example ex = pData->getExample( results[i]->getIdx() );
				vector<Label> labs = ex.getLabels();
				double m = numeric_limits<double>::infinity();
				for (int l = 0; l < numClasses; ++l)  
				{
					if ( labs[l].y > 0 )
					{
						if ( expVotesForExamples[l] > numeric_limits<double>::epsilon() )
						{
							double logVal = log( expVotesForExamples[l] );
							
							if ( logVal != m ) {
								lLambda += ( ( 1.0/(double)numExamples ) * logVal );
							}
						}
					}
				}


			}
			

			outFile << t << "\t" << lLambda ;
			outFile << '\n';
			
			outFile.flush();
		}

		if (pOutInfo)
			delete pOutInfo;

		// computeResults( pData, weakHypotheses, results, numIterations );
		///////////////////////////////////////////////////////////////////////////////////


		/*
		for (int i = 0; i < numExamples; ++i)
		{
			// output the name if it exists, otherwise the number
			// of the example
			exampleName = pData->getExampleName(i);
			if ( !exampleName.empty() )
				outFile << exampleName << ',';

			// output the posteriors
			outFile << results[i]->getVotesVector()[0];
			for (int l = 1; l < numClasses; ++l)
				outFile << ',' << results[i]->getVotesVector()[l];
			outFile << '\n';
		}
		*/

		if (_verbose > 0)
			cout << "Done!" << endl;

		if (_verbose > 1)
		{
			cout << "\nClass order (You can change it in the header of the data file):" << endl;
			for (int l = 0; l < numClasses; ++l)
				cout << "- " << pData->getClassMap().getNameFromIdx(l) << endl;
		}

		// delete the input data file
		if (pData) 
			delete pData;

		vector<ExampleResults*>::iterator it;
		for (it = results.begin(); it != results.end(); ++it)
			delete (*it);
	}


	// -------------------------------------------------------------------------

	//void LogitBoostClassifier::saveSingleStumpFeatureData(const string& dataFileName, 
	//                                                      const string& shypFileName,
	//                                                      const string& outFileName, int numIterations)
	//{
	//   InputData* pData = loadInputData(dataFileName, shypFileName);
	//
	//   if (_verbose > 0)
	//      cout << "Loading strong hypothesis..." << flush;
	//
	//   // The class that loads the weak hypotheses
	//   UnSerialization us;
	//
	//   // Where to put the weak hypotheses
	//   vector<BaseLearner*> weakHypotheses;
	//
	//   // loads them
	//   us.loadHypotheses(shypFileName, weakHypotheses);
	//
	//   if (weakHypotheses.empty())
	//      return;
	//
	//   if (numIterations == 0)
	//      numIterations = static_cast<int>(weakHypotheses.size());
	//
	//   if (_verbose > 0)
	//      cout << "Creating file..." << flush;
	//
	//   const int numClasses = ClassMappings::getNumClasses();
	//   const int numExamples = pData->getNumExamples();
	//
	//   vector<BaseLearner*>::const_iterator whyIt;
	//
	//   vector< vector<float> > data( numIterations );
	//   vector< string > classColumn;
	//   int pos = 0;
	//   int t;
	//
	//   // create the first column
	//   int numRows = numClasses+numExamples;
	//   classColumn.resize(numRows);
	//   for (int l = 0; l < numClasses; ++l)
	//      classColumn[pos++] = "0";
	//   for (int i = 0; i < numExamples; ++i)
	//      classColumn[pos++] = ClassMappings::getClassNameFromIdx( pData->getClass(i) );
	//
	//   // get the data for the other columns
	//   for (t = 0, whyIt = weakHypotheses.begin(); 
	//        t < numIterations && whyIt != weakHypotheses.end(); 
	//        ++whyIt, ++t)
	//   {
	//      BaseLearner* currWeakHyp = *whyIt;
	//      vector<float>& tmpCol = data[t];
	//
	//      currWeakHyp->getStateData(tmpCol, "ssfeatures", pData);
	//   }
	//
	//   if ( data[1].empty() )
	//   {
	//      cerr << "ERROR: option -ssfeatures works only with SingleStump feature types!" << endl;
	//      exit(1);
	//   }
	//
	//
	//   //////////////////////////////////////////////////////////////////////////
	//   ofstream outFile(outFileName.c_str());
	//
	//   // Print alpha!
	//   // for every feature: 1..numIterations
	//   outFile << "0\t";
	//   for (t = 0, whyIt = weakHypotheses.begin(); 
	//        t < numIterations && whyIt != weakHypotheses.end(); 
	//        ++whyIt, ++t)
	//   {
	//      BaseLearner* currWeakHyp = *whyIt;
	//      outFile << currWeakHyp->getAlpha() << '\t';
	//   }
	//   outFile << endl;
	//
	//   // now print all the data
	//   for (int r = 0; r < numRows; ++r)
	//   {
	//      outFile << classColumn[r];
	//      for (t = 0; t < numIterations; ++t)
	//         outFile << '\t' << data[t][r];
	//      outFile << '\n';
	//   }
	//
	//   //////////////////////////////////////////////////////////////////////////
	//
	//   if (_verbose > 0)
	//      cout << "Done!" << endl;
	//
	//   // delete the input data file
	//   if (pData) 
	//      delete pData;
	//}


	// -------------------------------------------------------------------------

	InputData* LogitBoostClassifier::loadInputData(const string& dataFileName, const string& shypFileName)
	{
		// open file
		ifstream inFile(shypFileName.c_str());
		if (!inFile.is_open())
		{
			cerr << "ERROR: Cannot open strong hypothesis file <" << shypFileName << ">!" << endl;
			exit(1);
		}

		// Declares the stream tokenizer
		nor_utils::StreamTokenizer st(inFile, "<>\n\r\t");

		// Move until it finds the multiboost tag
		if ( !UnSerialization::seekSimpleTag(st, "multiboost") )
		{
			// no multiboost tag found: this is not the correct file!
			cerr << "ERROR: Not a valid MultiBoost Strong Hypothesis file!!" << endl;
			exit(1);
		}

		// Move until it finds the algo tag
		string basicLearnerName = UnSerialization::seekAndParseEnclosedValue<string>(st, "algo");

		// Check if the weak learner exists
		if ( !BaseLearner::RegisteredLearners().hasLearner(basicLearnerName) )
		{
			cerr << "ERROR: Weak learner <" << basicLearnerName << "> not registered!!" << endl;
			exit(1);
		}

		// get the training input data, and load it
		BaseLearner* baseLearner = BaseLearner::RegisteredLearners().getLearner(basicLearnerName);
		baseLearner->initLearningOptions(_args);
		InputData* pData = baseLearner->createInputData();

		// set the non-default arguments of the input data
		pData->initOptions(_args);
		// load the data
		pData->load(dataFileName, IT_TEST, _verbose);

		return pData;
	}

	// -------------------------------------------------------------------------

	// Returns the results into ptRes
	void LogitBoostClassifier::computeResults(InputData* pData, vector<BaseLearner*>& weakHypotheses, 
		vector< ExampleResults* >& results, int numIterations)
	{
		assert( !weakHypotheses.empty() );

		const int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();

		// Initialize the output info
		OutputInfo* pOutInfo = NULL;

		if ( !_outputInfoFile.empty() )
			pOutInfo = new OutputInfo(_outputInfoFile);

		// Creating the results structures. See file Structures.h for the
		// PointResults structure
		results.clear();
		results.reserve(numExamples);
		for (int i = 0; i < numExamples; ++i)
			results.push_back( new ExampleResults(i, numClasses) );

		// iterator over all the weak hypotheses
		vector<BaseLearner*>::const_iterator whyIt;
		int t;

		pOutInfo->initialize( pData );

		// for every feature: 1..T
		for (whyIt = weakHypotheses.begin(), t = 0; 
			whyIt != weakHypotheses.end() && t < numIterations; ++whyIt, ++t)
		{
			BaseLearner* currWeakHyp = *whyIt;
			float alpha = currWeakHyp->getAlpha();

			// for every point
			for (int i = 0; i < numExamples; ++i)
			{
				// a reference for clarity and speed
				vector<float>& currVotesVector = results[i]->getVotesVector();

				// for every class
				for (int l = 0; l < numClasses; ++l)
					currVotesVector[l] += alpha * currWeakHyp->classify(pData, i, l);
			}

			// if needed output the step-by-step information
			if ( pOutInfo )
			{
				pOutInfo->outputIteration(t);
				pOutInfo->outputError(pData, currWeakHyp);
				
				pOutInfo->outputBalancedError(pData, currWeakHyp);
				if ( ( t % 1 ) == 0 ) {
					pOutInfo->outputROC(pData, currWeakHyp);
				}

				// Margins and edge requires an update of the weight,
				// therefore I keep them out for the moment
				//outInfo.outputMargins(pData, currWeakHyp);
				//outInfo.outputEdge(pData, currWeakHyp);
				pOutInfo->endLine();
			}
		}

		if (pOutInfo)
			delete pOutInfo;

	}

	// -------------------------------------------------------------------------

	float LogitBoostClassifier::getOverallError( InputData* pData, const vector<ExampleResults*>& results, 
		int atLeastRank )
	{
		const int numExamples = pData->getNumExamples();
		int numErrors = 0;

		assert(atLeastRank >= 0);

		for (int i = 0; i < numExamples; ++i)
		{
			// if the actual class is not the one with the highest vote in the
			// vote vector, then it is an error!
			if ( !results[i]->isWinner( pData->getExample(i), atLeastRank ) )
				++numErrors;
		}  

		// makes the error between 0 and 1
		return (float)numErrors / (float)numExamples;
	}

	// -------------------------------------------------------------------------

	void LogitBoostClassifier::getClassError( InputData* pData, const vector<ExampleResults*>& results, 
		vector<float>& classError, int atLeastRank )
	{
		const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();

		classError.resize( numClasses, 0 );

		assert(atLeastRank >= 0);

		for (int i = 0; i < numExamples; ++i)
		{
			// if the actual class is not the one with the highest vote in the
			// vote vector, then it is an error!
			if ( !results[i]->isWinner( pData->getExample(i), atLeastRank ) )
			{
				const vector<Label>& labels = pData->getLabels(i);
				vector<Label>::const_iterator lIt;
				for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
				{
					if ( lIt->y > 0 )
						++classError[ lIt->idx ];
				}
				//++classError[ pData->getClass(i) ];
			}
		}

		// makes the error between 0 and 1
		for (int l = 0; l < numClasses; ++l)
			classError[l] /= (float)pData->getNumExamplesPerClass(l);
	}

	// -------------------------------------------------------------------------

} // end of namespace shogun
