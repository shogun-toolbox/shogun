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
#include "classifier/boosting/Classifiers/ABMHClassifierYahoo.h"
#include "classifier/boosting/Classifiers/ExampleResults.h"

#include "classifier/boosting/WeakLearners/SingleStumpLearner.h" // for saveSingleStumpFeatureData

#include <iomanip> // for setw
#include <cmath> // for pow
#include <functional>
#include <algorithm>

namespace MultiBoost {

	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------

	ABMHClassifierYahoo::ABMHClassifierYahoo(const nor_utils::Args &args, int verbose)
		: AdaBoostMHClassifier( args, verbose ), _queryFile( "" ), _queryIDSBorders(0), _origLabels(0), _labels(0)
	{
		// The file with the step-by-step information
		if ( args.hasArgument("queryfile") )
			args.getValue("queryfile", 0, _queryFile );

		string scoringType;
		if ( args.hasArgument("scoring") )
			args.getValue("scoring", 0, scoringType );

		if ( scoringType.compare( "fourthclass" ) == 0 )
			_scoring = EVAL_FOURTH_LABEL;
		else if ( scoringType.compare( "expweight" ) == 0 )
			_scoring = EVAL_EXP_WEIGHT;
		else {
			cerr << "Unknown scoring (Exponential weighting)" << endl;
			_scoring = EVAL_EXP_WEIGHT;
		}

	}

	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------

	// Returns the results into ptRes
	void ABMHClassifierYahoo::computeResults(InputData* pData, vector<BaseLearner*>& weakHypotheses, 
		vector< ExampleResults* >& results, vector< ExampleResults* >& normalizedResults, int numIterations)
	{
		assert( !weakHypotheses.empty() );

		const int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();

		vector<Label>::const_iterator lIt;

		//read query file
		this->readQueries();

		// Initialize the output info
		OutputInfo* pOutInfo = NULL;

		if ( !_outputInfoFile.empty() )
			pOutInfo = new OutputInfo(_outputInfoFile);

		// Creating the results structures. See file Structures.h for the
		// PointResults structure
		results.clear();
		results.reserve(numExamples);
		normalizedResults.clear();
		normalizedResults.reserve(numExamples);
		for (int i = 0; i < numExamples; ++i)
		{
			results.push_back( new ExampleResults(i, numClasses) );
			normalizedResults.push_back( new ExampleResults(i, numClasses) );
		}

		// iterator over all the weak hypotheses
		vector<BaseLearner*>::const_iterator whyIt;
		int t;
		float err = 0.0;
		float ncdg = 0.0;
		float sumAlpha = 0.0;
		if ( pOutInfo )
			pOutInfo->initialize( pData );

		// for every feature: 1..T
		for (whyIt = weakHypotheses.begin(), t = 0; 
			whyIt != weakHypotheses.end() && t < numIterations; ++whyIt, ++t)
		{
			BaseLearner* currWeakHyp = *whyIt;
			float alpha = currWeakHyp->getAlpha();
			sumAlpha += alpha;

			// for every point
			float mn = numeric_limits<float>::max();
			for (int i = 0; i < numExamples; ++i)
			{
				// a reference for clarity and speed
				vector<float>& currVotesVector = results[i]->getVotesVector();

				// for every class
				for (int l = 0; l < numClasses; ++l) {
					currVotesVector[l] += alpha * currWeakHyp->classify(pData, i, l);
					if ( currVotesVector[l] < mn ) mn = currVotesVector[l];
				}
			}

			//normalizing votes
			/*
			vector<float> colMin( numClasses );
			vector<float> colMax( numClasses );
			fill( colMin.begin(), colMin.end(), numeric_limits<float>::max() );
			fill( colMax.begin(), colMax.end(), -numeric_limits<float>::max() );

			for (int i = 0; i < numExamples; ++i)
			{
				// a reference for clarity and speed
				vector<float>& currVotesVector = results[i]->getVotesVector();				

				vector<float>& currNormalizedVotesVector = normalizedResults[i]->getVotesVector();

				for (int l = 0; l < numClasses; ++l) {
					currNormalizedVotesVector[l] = currVotesVector[l] / sumAlpha;
				}


				for (int l = 0; l < numClasses; ++l) {
					if ( currNormalizedVotesVector[l] < colMin[l] ) colMin[l] = currNormalizedVotesVector[l];
					if ( currNormalizedVotesVector[l] > colMax[l] ) colMax[l] = currNormalizedVotesVector[l];
				}
			}

			for (int l = 0; l < numClasses; ++l) {
				colMax[l] -= colMin[l];
			}

			for (int i = 0; i < numExamples; ++i)
			{
				// a reference for clarity and speed
				vector<float>& currNormalizedVotesVector = normalizedResults[i]->getVotesVector();
				float sum = 0.0;
				for (int l = 0; l < numClasses; ++l) {
					currNormalizedVotesVector[l] = ( currNormalizedVotesVector[l] - colMin[l] ) / colMax[l];
					sum += currNormalizedVotesVector[l];
				}
				for (int l = 0; l < numClasses; ++l) {
					currNormalizedVotesVector[l] /= sum;
				}

			}			
			*/
			
			// kivonva a minimum es soronkent normalva
			
			for (int i = 0; i < numExamples; ++i)
			{
				// a reference for clarity and speed
				vector<float>& currVotesVector = results[i]->getVotesVector();
				
				float sum = 0.0;
				// for every class
				for (int l = 0; l < numClasses; ++l) {
					sum += ( ( currVotesVector[l] / sumAlpha ) - mn);
				}
				vector<float>& currNormalizedVotesVector = normalizedResults[i]->getVotesVector();

				for (int l = 0; l < numClasses; ++l) {
					currNormalizedVotesVector[l] = ( ( currVotesVector[l] / sumAlpha ) - mn ) / sum;
				}
			}
			
			//normalizing votes

			//
			for( int i=0; i < _queryIDSBorders.size(); i++ )
			{
				// collect the class labels for one query
				vector<int> origLabelPerDoc(0);
				vector<float> predValues(0);
				int leftBound;

				if ( i==0 )
					leftBound = 0;
				else 
					leftBound = _queryIDSBorders[i-1];

				for( int j=leftBound; j<_queryIDSBorders[i]; j++ )
				{

					// a reference for clarity and speed, predicted values
					//vector<float>& currVotesVector = results[j]->getVotesVector();
					vector<float>& currVotesVector = normalizedResults[j]->getVotesVector();
					
					const vector<Label>& labels = pData->getLabels(j);
					int labelIDX = -1;
					for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
					{
						if ( lIt->y > 0 ) labelIDX = lIt->idx;
					}
					
					//if ( ( !_labels.empty() ) && ( labelIDX != _labels[j] ) ) cout << "Warning: incorrect labels!!!" << endl;

					if ( ! _origLabels.empty() )
						origLabelPerDoc.push_back( _origLabels[j] );
					else
						origLabelPerDoc.push_back( 0 );

					float tmpScore = 0.0;
					switch (_scoring)
					{
					case EVAL_FOURTH_LABEL:
						predValues.push_back( currVotesVector[numClasses-1] );
						break;
					case EVAL_EXP_WEIGHT:						
						for( int j=1; j<numClasses; j++ )
						{
							tmpScore += currVotesVector[j] * pow( 2.0, j );
						}
						predValues.push_back( tmpScore );
						break;
					default:
						cerr << "Bad scoring!" << endl;
						exit( -1 );
					}
				}

				vector<int> ranks(0);
				normalizePredictedScores( predValues, ranks );

				float tmpErr, tmpncdg;				
				getERR( origLabelPerDoc, ranks, tmpErr, tmpncdg );

				err += tmpErr;
				ncdg += tmpncdg;
			}


			err /= ((float) _queryIDSBorders.size());
			ncdg /= ((float) _queryIDSBorders.size());

			// if needed output the step-by-step information
			if ( pOutInfo )
			{
				pOutInfo->outputIteration(t);
				pOutInfo->outputError(pData, currWeakHyp);

				//pOutInfo->outputBalancedError(pData, currWeakHyp);

				pOutInfo->outUserData( err );
				pOutInfo->outUserData( ncdg );
				/*
				if ( ( t % 1 ) == 0 ) {
				pOutInfo->outputROC(pData, currWeakHyp);
				}
				*/

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

	void ABMHClassifierYahoo::getERR( vector<int>& l, vector<int>& r, float& err, float &ndcg, int k )
	{
		err = 0.0;
		ndcg = 0.0;

		assert( l.size() == r.size() );
		int nd = r.size();

		// the rank smalller than the number of documents
		int mx = *max_element( r.begin(), r.end() );
		assert( mx <= r.size() );

		vector<float> gains( r.size() );
		fill( gains.begin(), gains.end(), -1 );

		for( int i=0; i<nd; i++ )
		{
			gains[r[i]-1]= ( pow( 2.0, l[i] ) - 1.0 ) / 16;
		}

		// all rank is presented
		float mn = *min_element( gains.begin(), gains.end() );
		assert( mn >= 0.0 );

		float p = 1.0;
		for( int i=0; i< nd; i++ )
		{
			float tmpR = gains[i];
			err += (p*tmpR)/(i+1.0);
			p *= (1.0-tmpR);
		}

		//ncdg
		float dcg = 0.0;
		float ideal_dcg = 0.0;

		for( int i=0; i<(k<nd?k:nd); i++ )
		{
			dcg+=(gains[i]/log(i+2.0)); 
		}
		sort( gains.begin(), gains.end(), greater<float>() );

		for( int i=0; i<(k<nd?k:nd); i++ )
		{
			ideal_dcg+=(gains[i]/log(i+2.0)); 
		}

		if ( !nor_utils::is_zero(ideal_dcg ) )
			ndcg += dcg / ideal_dcg;
		else
			ndcg += 0.5;


	}
	// -------------------------------------------------------------------------

	void ABMHClassifierYahoo::normalizePredictedScores( vector<float>& r, vector<int>& ranks )
	{
		vector<pair<float,int> > tmpArr( r.size() );
		for( int i=0; i<r.size(); ++i ) 
		{
			tmpArr[i].first = r[i];
			tmpArr[i].second = i;
		}
		sort( tmpArr.begin(), tmpArr.end(), nor_utils::comparePair<1, float, int, greater<float> >() );

		ranks.resize( r.size() );
		for( int i=0; i<r.size(); ++i ) 
		{
			ranks[tmpArr[i].second] = i+1;
		}
	}


	// -------------------------------------------------------------------------
	void ABMHClassifierYahoo::readQueries()
	{
		// open file
		ifstream inFile(_queryFile.c_str());
		if (!inFile.is_open())
		{
			cerr << "ERROR: Cannot open query file <" << _queryFile << ">!" << endl;
			exit(1);
		}

		//queryIDS.clear();
		_queryIDSBorders.clear();
		_origLabels.clear();
		_labels.clear();

		int currID=0, prevID=0, labOrig, lab;
		int i=0;
		string tmpLine;
		while( !inFile.eof() )
		{
			getline( inFile, tmpLine );
			stringstream ss(tmpLine);
			ss >> currID;
			if ( ss >> labOrig )
				_origLabels.push_back( labOrig );		
			if ( ss >> lab )
				_labels.push_back( lab );
			//cout<< currID;

			//queryIDS.push_back( currID );
			if ( (i>0) && ( prevID != currID ) )
				_queryIDSBorders.push_back( i );

			prevID = currID;
			i++;
		}
		_queryIDSBorders.push_back( --i );
		inFile.close();
	}

	// -------------------------------------------------------------------------

	void ABMHClassifierYahoo::run(const string& dataFileName, const string& shypFileName, 
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
		vector< ExampleResults* > normalziedResults;

		if (_verbose > 0)
			cout << "Classifying..." << flush;

		// get the results
		computeResults( pData, weakHypotheses, results, normalziedResults, numIterations );

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
			const int numClasses = pData->getNumClasses();

			ofstream outRes(outResFileName.c_str());
			//
			for( int i=0; i < _queryIDSBorders.size(); i++ )
			{
				// collect the class labels for one query
				vector<float> predValues(0);
				int leftBound;

				if ( i==0 )
					leftBound = 0;
				else 
					leftBound = _queryIDSBorders[i-1];

				for( int j=leftBound; j<_queryIDSBorders[i]; j++ )
				{

					// a reference for clarity and speed, predicted values
					vector<float>& currVotesVector = normalziedResults[j]->getVotesVector();
					float tmpScore = 0.0;
					switch (_scoring)
					{
					case EVAL_FOURTH_LABEL:
						predValues.push_back( currVotesVector[numClasses-1] );
						break;
					case EVAL_EXP_WEIGHT:						
						for( int j=1; j<numClasses; j++ )
						{
							tmpScore += currVotesVector[j] * pow( 2.0, j );
						}
						predValues.push_back( tmpScore );
						break;
					default:
						cerr << "Bad scoring!" << endl;
						exit( -1 );
					}
				}

				vector<int> ranks(0);
				normalizePredictedScores( predValues, ranks );

				for( int i=0; i<ranks.size()-1; i++ )
				{
					outRes << ranks[i] << " ";
				}
				outRes << ranks[ranks.size()-1] << endl;
			}

			outRes.close();
			if (_verbose > 0)
				cout << "\nPredictions written on file <" << outResFileName << ">!" << endl;

			//output the posteriors
			string posteriorFilename = outResFileName;
			posteriorFilename.append( ".pos" );
			ofstream outPost( posteriorFilename.c_str() );
			
			string exampleName;
			
			for (int i = 0; i < numExamples; ++i)
			{
				// output the name if it exists, otherwise the number
				// of the example
				/*
				exampleName = pData->getExampleName(i);
				if ( exampleName.empty() )
					outPost << i << '\t';
				else
					outPost << exampleName << '\t';
				*/
				// output the predicted class
				//outRes << pData->getClassMap().getNameFromIdx( results[i]->getWinner().first ) << endl;

				vector<float>& currVotesVector = results[i]->getVotesVector();
				//const vector<Label>& labels = pData->getLabels(i);
				for( int j=0; j<numClasses; j++ )
				{
					outPost << currVotesVector[j] << "\t";
				}
				outPost << endl;
			}
			
			outPost.close();


			if (_verbose > 0)
				cout << "\nPosteriors written on file <" << posteriorFilename << ">!" << endl;

		}


		// delete the input data file
		if (pData) 
			delete pData;

		vector<ExampleResults*>::iterator it;
		for (it = results.begin(); it != results.end(); ++it)
			delete (*it);
		if (_verbose > 0)
			cout << "Ready!" << endl;

	}
	// -------------------------------------------------------------------------

} // end of namespace MultiBoost
