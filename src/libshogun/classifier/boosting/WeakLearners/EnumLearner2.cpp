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


#include "EnumLearner2.h"
#include <limits>
#include <cstdlib>
#include "classifier/boosting/Kmeans/KMlocal.h"			// k-means algorithms
#include "classifier/boosting/IO/Serialization.h"

namespace shogun {

	//REGISTER_LEARNER_NAME(SingleStump, EnumLearner2)
	REGISTER_LEARNER(EnumLearner2)

	//coantins the previosly selected U vector, if this member is empty then the run function hasn't been valled before
    //vector<float> EnumLearner2::_prevU(0);
	vector< int > EnumLearner2::_clusters( 0 );
	int EnumLearner2::_k = 3;
		// ------------------------------------------------------------------------------

		float EnumLearner2::run()
	{
		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();
		const int numExamples = _pTrainingData->getNumExamples();
		
		if ( EnumLearner2::_clusters.empty() ) EnumLearner2::generateSimilarityMatrix();

		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

		vector<sRates> vMu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<float> tmpV(numClasses); // The class-wise votes/abstentions
		vector<float> previousTmpV(numClasses); // The class-wise votes/abstentions

		float tmpAlpha,previousTmpAlpha, previousEnergy;
		float bestEnergy = numeric_limits<float>::max();

		int numOfDimensions = _maxNumOfDimensions;
		for (int j = 0; j < numColumns; ++j)
		{
			// Tricky way to select numOfDimensions columns randomly out of numColumns
			int rest = numColumns - j;
			float r = rand()/static_cast<float>(RAND_MAX);

			if ( static_cast<float>(numOfDimensions) / rest > r ) 
			{
				--numOfDimensions;

				if (_verbose > 2)
					cout << "    --> trying attribute = "
					<<_pTrainingData->getAttributeNameMap().getNameFromIdx(j)
					<< endl << flush;

				const int numIdxs = _pTrainingData->getEnumMap(j).getNumNames();

				// Create and initialize the numIdxs x numClasses gamma matrix
				vector<vector<float> > tmpGammasPls(numIdxs);
				vector<vector<float> > tmpGammasMin(numIdxs);
				for (int io = 0; io < numIdxs; ++io) {
					vector<float> tmpGammaPls(numClasses);
					vector<float> tmpGammaMin(numClasses);
					fill(tmpGammaPls.begin(), tmpGammaPls.end(), 0.0);
					fill(tmpGammaMin.begin(), tmpGammaMin.end(), 0.0);
					tmpGammasPls[io] = tmpGammaPls;
					tmpGammasMin[io] = tmpGammaMin;
				}

				// Compute the elements of the gamma plus and minus matrices
				float entry;
				for (int i = 0; i < numExamples; ++i) {
					const vector<Label>& labels = _pTrainingData->getLabels(i);
					int io = static_cast<int>(_pTrainingData->getValue(i,j));	    
					for (int l = 0; l < numClasses; ++l) {
						entry = labels[l].weight * labels[l].y;
						if (entry > 0)
							tmpGammasPls[io][l] += entry;
						else if (entry < 0)
							tmpGammasMin[io][l] += -entry;
					}
				}

				// Initialize the u vector to random +-1
				vector<sRates> uMu(numIdxs); // The idx-wise rates
				vector<float> tmpU(numIdxs);// The idx-wise votes/abstentions
				vector<float> previousTmpU(numIdxs);// The idx-wise votes/abstentions
				
								
				for (int io = 0; io < numIdxs; ++io) {
					uMu[io].classIdx = io;	    
					if ( rand()/static_cast<float>(RAND_MAX) > 0.5 )
						tmpU[io] = +1;
					else
						tmpU[io] = -1;
				}
				
				//instead of random initialization, we set the similar user equal to each other
				
				int clusterIdx = (int) ( rand()/static_cast<float>(RAND_MAX) * (_k - 1) );
				
				for( int io = 0; io < numIdxs; io++  ) {
					if ( EnumLearner2::_clusters[io] == clusterIdx ) 
						tmpU[io] = +1;
					
					else
						tmpU[io] = -1;
				}
				
				/*
				for (int io = 0; io < numIdxs; ++io) {
					cout << tmpU[io] << " ";
				}
				cout << endl;
				*/

				vector<sRates> vMu(numClasses); // The label-wise rates
				for (int l = 0; l < numClasses; ++l)
					vMu[l].classIdx = l;
				vector<float> tmpV(numClasses); // The label-wise votes/abstentions

				float tmpEnergy = numeric_limits<float>::max();
				float tmpVal;
				tmpAlpha = 0.0;

				while (1) {
					previousEnergy = tmpEnergy;
					previousTmpV = tmpV;
					previousTmpAlpha = tmpAlpha;

					//filling out tmpV and vMu
					for (int l = 0; l < numClasses; ++l) {
						vMu[l].rPls = vMu[l].rMin = vMu[l].rZero = 0; 
						for (int io = 0; io < numIdxs; ++io) {
							if (tmpU[io] > 0) {
								vMu[l].rPls += tmpGammasPls[io][l];
								vMu[l].rMin += tmpGammasMin[io][l];
							}
							else if (tmpU[io] < 0) {
								vMu[l].rPls += tmpGammasMin[io][l];
								vMu[l].rMin += tmpGammasPls[io][l];
							}
						}
						if (vMu[l].rPls >= vMu[l].rMin) {
							tmpV[l] = +1;
						}
						else {
							tmpV[l] = -1;
							tmpVal = vMu[l].rPls;
							vMu[l].rPls = vMu[l].rMin;
							vMu[l].rMin = tmpVal;
						}
					}

					tmpEnergy = AbstainableLearner::getEnergy(vMu, tmpAlpha, tmpV);

					if (_verbose > 2)
						cout << "        --> energy V = " << tmpEnergy << "\talpha = " << tmpAlpha << endl << flush;

					if (tmpEnergy >= previousEnergy) {
						tmpV = previousTmpV;
						break;
					}

					previousEnergy = tmpEnergy;
					previousTmpU = tmpU;
					previousTmpAlpha = tmpAlpha;

					//filling out tmpU and uMu
					for (int io = 0; io < numIdxs; ++io) {
						uMu[io].rPls = uMu[io].rMin = uMu[io].rZero = 0; 
						for (int l = 0; l < numClasses; ++l) {
							if (tmpV[l] > 0) {
								uMu[io].rPls += tmpGammasPls[io][l];
								uMu[io].rMin += tmpGammasMin[io][l];
							}
							else if (tmpV[l] < 0) {
								uMu[io].rPls += tmpGammasMin[io][l];
								uMu[io].rMin += tmpGammasPls[io][l];
							}
						}
						if (uMu[io].rPls >= uMu[io].rMin) {
							tmpU[io] = +1;
						}
						else {
							tmpU[io] = -1;
							tmpVal = uMu[io].rPls;
							uMu[io].rPls = uMu[io].rMin;
							uMu[io].rMin = tmpVal;
						}
					}

					tmpEnergy = AbstainableLearner::getEnergy(uMu, tmpAlpha, tmpU);

					if (_verbose > 2)
						cout << "        --> energy U = " << tmpEnergy << "\talpha = " << tmpAlpha << endl << flush;

					if (tmpEnergy >= previousEnergy) {
						tmpU = previousTmpU;
						break;
					}
				}

				if ( previousEnergy < bestEnergy && previousTmpAlpha > 0 ) {
					_alpha = previousTmpAlpha;
					_v = tmpV;
					_u = tmpU;
					_selectedColumn = j;
					bestEnergy = previousEnergy;
				}
			}
		}


		_id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn);
		return bestEnergy;

	}

	// ------------------------------------------------------------------------------

	void EnumLearner2::generateSimilarityMatrix( void )
	{
		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();
		const int numExamples = _pTrainingData->getNumExamples();

		// execution parameters (see KMterm.h and KMlocal.h)
		KMterm  term(100, 0, 0, 0,              // run for 100 stages
					 0.10, 0.10, 3,             // other typical parameter values 
					 0.50, 10, 0.95);
		
		const int mostSimilarK = 5;

		if ( numColumns != 2 ) 
		{
			cout << "For collaborating filtering the databeas has to contain two columns!!!!!" << endl;
			exit( -1 );
		}


		
		const int userNum = _pTrainingData->getEnumMap( 0 ).getNumNames();
		const int objectNum = _pTrainingData->getEnumMap( 1 ).getNumNames();

		if ( _verbose > 0 ) {
			cout << "Allocate memory for vote matrix...";
		}

		KMdata dataPts(objectNum, userNum);
		vector< map< int, float> > voteMatrix( userNum );
		
		/*
		_similarityMatrix.resize( userNum );
		
		
		vector< priority_queue< pair< float, int > > > simMatrix( userNum );
		for( int i = 0; i < userNum; i++ ) {
			priority_queue< pair<float, int> > tmpPq;
			simMatrix[i] = tmpPq;
		}
		*/


		if ( _verbose > 0 ) {
			cout << "Ready!" << endl;
		}

		for( int i = 0; i<numExamples; i++ ) 
		{
			int userId =  (int) _pTrainingData->getValue( i, 0 );
			int objectId = (int) _pTrainingData->getValue( i, 1 );
			vector< Label > labs = _pTrainingData->getLabels( i );
			for( int j = 0; j < numClasses; j++ ) 
			{
				//if ( _pTrainingData->hasLabel( i, j ) ) cout << j << endl;
				if ( labs[j].y == 1 ) {
					//cout << j << endl;
					voteMatrix[userId].insert( map<int,float>::value_type( objectId, (float) j ) );
				}
			}

			//cout << userId << " " << objectId << endl;			
		}
		
		
		KMdataArray pa = dataPts.getPts();
		for( int i = 0; i<userNum; i++ ) {
			for( int j = 0; j<objectNum; j++ ) {
				map< int, float >::iterator it = voteMatrix[i].find( j );
				if ( it != voteMatrix[i].end() ) {
					pa[i][j] = (KMcoord) it->second;
				} else {
					pa[i][j] = (KMcoord) 0.0;
				}
			}
		}

		if ( _verbose > 0 ) { 
			cout << "Clustering..." << endl;
		}

		dataPts.buildKcTree();			// build filtering structure
		KMfilterCenters ctrs(_k, dataPts);		// allocate centers

    							// run the algorithm
		KMlocalLloyds       kmAlg(ctrs, term);	// repeated Lloyd's
		// KMlocalSwap      kmAlg(ctrs, term);	// Swap heuristic
		// KMlocalEZ_Hybrid kmAlg(ctrs, term);	// EZ-Hybrid heuristic
		// KMlocalHybrid    kmAlg(ctrs, term);	// Hybrid heuristic
		

		ctrs = kmAlg.execute();			// execute

		if ( _verbose > 0 ) { 
			cout << "Ready with the clustering..." << endl;
		}
		
		KMctrIdxArray closeCtr = new KMctrIdx[dataPts.getNPts()];
		double* sqDist = new double[dataPts.getNPts()];
		ctrs.getAssignments(closeCtr, sqDist);

		EnumLearner2::_clusters.resize(dataPts.getNPts() );
		
		vector< int > numElementsPerCluster( EnumLearner2::_k );
		fill( numElementsPerCluster.begin(), numElementsPerCluster.end(), 0 );

		for( int i = 0; i < dataPts.getNPts(); i++ ) {
			//cout << closeCtr[i] << " ";
			EnumLearner2::_clusters[i] = closeCtr[i];
			numElementsPerCluster[EnumLearner2::_clusters[i]]++;
		}

		for( int i = 0; i < numElementsPerCluster.size(); i++ ) {
			cout << "Cluster " << i << " " << numElementsPerCluster[i] << endl;
		}

		//cout << endl;

		delete [] closeCtr;
		delete [] sqDist;


		/*
		for( int i = 0; i < userNum - 1; i++ )  {
			for( int j = i+1; j<userNum; j++ ) {
				
				float similarity = 0.0;
				int voteNum = 0;
				for( map<int,float>::iterator it1 = voteMatrix[i].begin(); it1 != voteMatrix[i].end(); it1++ ) {
					
					map< int, float >::iterator it2 = voteMatrix[j].find( it1->first );
					if ( it2 != voteMatrix[j].end() ) {
						similarity += ( ( it1->second - it2->second ) * ( it1->second - it2->second ) );
						voteNum++;
					}
				}

				if ( voteNum > 0 ) { 
					similarity /= (float)voteNum;
					similarity = -similarity;

					pair< float, int > tmpPair1( similarity, j );
					simMatrix[i].push( tmpPair1 );

					pair< float, int > tmpPair2( similarity, i );
					simMatrix[j].push( tmpPair2 );
				}
			}
		}


		
		for( int i = 0; i<userNum; i++ )  {
			size_t st = mostSimilarK;
			if ( simMatrix[i].size() < mostSimilarK ) st = simMatrix[i].size();

			_similarityMatrix[i].resize( st );
			int j = 0;
			float sum = 0.0;
			while ( ( ! simMatrix[i].empty() ) && ( j < mostSimilarK ) ) {
				pair< float, int > tmpPair = simMatrix[i].top();
				simMatrix[i].pop();

				_similarityMatrix[i][j].first = tmpPair.second;
				_similarityMatrix[i][j].second = -tmpPair.first;
				
				sum += _similarityMatrix[i][j].second;
				j++;
			}

			for( j=0; j < _similarityMatrix[i].size(); j++ ) _similarityMatrix[i][j].second /= sum;
		}

    	//write out the similarity matrix
		ofstream out;
		out.open( "sim.txt" );
		
		for( int i = 0; i<userNum; i++ )  {
			//out << i << endl;
			for( int j=0; j < _similarityMatrix[i].size(); j++ )  out << _similarityMatrix[i][j].first << " ";
			out << endl;
			for( int j=0; j < _similarityMatrix[i].size(); j++ )  out << _similarityMatrix[i][j].second << " ";
			out << endl;
			out.flush();
		}
		
		out.close();

		out.clear();

		out.open( "mat.txt" );
		
		for( int i = 0; i < userNum; i++ )  {
			for( int j = 0; j < objectNum; j++ )  {
				map< int, float >::iterator it = voteMatrix[i].find( j );
				if ( it != voteMatrix[i].end() ) {
					out << it->second << "\t";
				} else {
					out << "0.0" << "\t";
				}
			}
			out << endl;
		}

		out.close();
		*/

		//end of writing out the similarity matrx
		if ( _verbose > 0 ) {
			cout << "Similarity matrix ready..." << endl;
		}
	}

	// ------------------------------------------------------------------------------

	float EnumLearner2::phi(float val, int /*classIdx*/) const
	{
		return _u[static_cast<int>(val)];
	}

	// -----------------------------------------------------------------------

	void EnumLearner2::save(ofstream& outputStream, int numTabs)
	{
		// Calling the super-class method
		FeaturewiseLearner::save(outputStream, numTabs);

		// save the _u vector
		outputStream << Serialization::vectorTag("uArray", _u, 
			_pTrainingData->getEnumMap(_selectedColumn), 
			"idx", (float) 0.0, numTabs) << endl;
	}

	// -----------------------------------------------------------------------

	void EnumLearner2::load(nor_utils::StreamTokenizer& st)
	{
		// Calling the super-class method
		FeaturewiseLearner::load(st);

		// load phiArray data
		UnSerialization::seekAndParseVectorTag(st, "uArray", _pTrainingData->getEnumMap(_selectedColumn), 
			"idx", _u);
	}

	// -----------------------------------------------------------------------

	void EnumLearner2::subCopyState(BaseLearner *pBaseLearner)
	{
		FeaturewiseLearner::subCopyState(pBaseLearner);

		EnumLearner2* pEnumLearner2 =
			dynamic_cast<EnumLearner2*>(pBaseLearner);

		pEnumLearner2->_u = _u;
	}

} // end of namespace shogun
