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


#include "TreeLearner2.h"

#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/Others/Example.h"
#include "classifier/boosting/Utils/StreamTokenizer.h"

#include <cmath>
#include <limits>
#include <queue>

namespace shogun {

	//REGISTER_LEARNER_NAME(Product, TreeLearner2)
	REGISTER_LEARNER(TreeLearner2)

		// -----------------------------------------------------------------------

		void TreeLearner2::declareArguments(nor_utils::Args& args)
	{
		BaseLearner::declareArguments(args);

		args.declareArgument("baselearnertype", 
			"The name of the learner that serves as a basis for the product\n"
			"  and the number of base learners to be multiplied\n"
			"  Don't forget to add its parameters\n",
			2, "<baseLearnerType> <numBaseLearners>");

	}

	// ------------------------------------------------------------------------------

	void TreeLearner2::initLearningOptions(const nor_utils::Args& args)
	{
		BaseLearner::initLearningOptions(args);

		string baseLearnerName;
		args.getValue("baselearnertype", 0, baseLearnerName);   
		args.getValue("baselearnertype", 1, _numBaseLearners);   

		// get the registered weak learner (type from name)
		BaseLearner* pWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner(baseLearnerName);

		for( int ib = 0; ib < _numBaseLearners; ++ib ) {
			_baseLearners.push_back(pWeakHypothesisSource->create());
			_baseLearners[ib]->initLearningOptions(args);

			vector< int > tmpVector( 2, -1 );
			_idxPairs.push_back( tmpVector );
		}
	}

	// ------------------------------------------------------------------------------

	float TreeLearner2::classify(InputData* pData, int idx, int classIdx)
	{
		float result  = 1;
		int ib = 0;
		while ( 1 ) {
			float phix = _baseLearners[ib]->classify(pData,idx,0);
			if ( phix > 0 ) {
				if ( _idxPairs[ ib ][ 0 ] > 0 ) { 
					ib = _idxPairs[ ib ][ 0 ];
				} else {
					return _baseLearners[ib]->classify( pData, idx, classIdx ); 
				}
			} else {
				if ( _idxPairs[ ib ][ 1 ] > 0 ) { 
					ib = _idxPairs[ ib ][ 1 ];
				} else {
					return _baseLearners[ib]->classify( pData, idx, classIdx ); 
				}
			}
		}
	}

	//-------------------------------------------------------------------------------

	float TreeLearner2::getEdge( BaseLearner* learner, InputData* d ) {
		float edge = 0.0;
		for( int i = 0; i < d->getNumExamples(); i++ ) {

			vector< Label > l = d->getLabels( i );
			//cout << d->getRawIndex( i ) << " " << endl;

			for( vector<Label>::iterator it = l.begin(); it !=  l.end(); it++ ) {
				float cl = learner->classify( d, i, it->idx );
				edge += ( cl * it->weight * it->y );
			}
		}
		//cout << endl;
		return edge;
	}

	// ------------------------------------------------------------------------------

	float TreeLearner2::run()
	{
		for(int ib = 0; ib < _numBaseLearners; ++ib)
			_baseLearners[ib]->setTrainingData(_pTrainingData);

		float edge = numeric_limits<float>::max();

		BaseLearner* pPreviousBaseLearner = 0;
		set< int > tmpIdx, idxPos, idxNeg;
		floatBaseLearner tmpPair, tmpPairPos, tmpPairNeg;

		// for storing the inner point (learneres) which will be extended
		vector< floatBaseLearner > bLearnerVector;
		floatBaseLearnerVector innerNode;
		priority_queue<floatBaseLearnerVector, deque<floatBaseLearnerVector>, greater_first<floatBaseLearnerVector> > pq;

		//train the first learner
		//_baseLearners[0]->run();
		pPreviousBaseLearner = _baseLearners[0]->copyState();
		pPreviousBaseLearner->run();
		
		_pTrainingData->clearIndexSet();
		for( int i = 0; i < _pTrainingData->getNumExamples(); i++ ) tmpIdx.insert( i );

		//this contains the number of baselearners 
		int ib = 0;

		//set the edge
		tmpPair.first = getEdge( pPreviousBaseLearner, _pTrainingData );
		tmpPair.second.first.first = pPreviousBaseLearner;
		// set the pointer of the parent
		tmpPair.second.first.second.first = 0;
		// set that this is a neg child
		tmpPair.second.first.second.second = 0;
		tmpPair.second.second = tmpIdx;

		bLearnerVector = calculateChildrenAndEnergies( tmpPair );

		//insert the root into the priority queue

		if ( ! bLearnerVector.empty() ) 
		{
			if (_verbose > 2) {
				cout << "Edges: (parent, pos, neg): " << bLearnerVector[0].first << " " << bLearnerVector[1].first << " " << bLearnerVector[2].first << endl << flush;
				//cout << "alpha[" << (ib) <<  "] = " << _alpha << endl << flush;
			}

			// if the energy is getting higher then we push it into the priority queue
			if ( bLearnerVector[0].first < ( bLearnerVector[1].first + bLearnerVector[2].first ) ) {
				float deltaEdge = abs( 	bLearnerVector[0].first - ( bLearnerVector[1].first + bLearnerVector[2].first ) );
				innerNode.first = deltaEdge;
				innerNode.second = bLearnerVector;
				pq.push( innerNode ); 
			} else {
				//delete bLearnerVector[0].second.first.first;
				delete bLearnerVector[1].second.first.first;
				delete bLearnerVector[2].second.first.first;
			}
			
		}  
		
		if ( pq.empty() ) {
			// we don't extend the root			
			BaseLearner* tmpBL = _baseLearners[0];
			_baseLearners[0] = tmpPair.second.first.first;
			delete tmpBL;				
			ib = 1;
		}







		while ( ! pq.empty() && ( ib < _numBaseLearners ) ) {
			//get the best learner from the priority queue
			innerNode = pq.top();

			if (_verbose > 2) {
				cout << "Delta energy: " << innerNode.first << endl << flush;
				cout << "Size of priority queue: " << pq.size() << endl << flush;
			}

			pq.pop();
			bLearnerVector = innerNode.second;

			tmpPair = bLearnerVector[0];
			tmpPairPos = bLearnerVector[1];
			tmpPairNeg = bLearnerVector[2];

			//store the baselearner if the deltaenrgy will be higher
			if ( _verbose > 3 ) {
				cout << "Insert learner: " << ib << endl << flush;
			}
			int parentIdx = tmpPair.second.first.second.first;
			BaseLearner* tmpBL = _baseLearners[ib];
			_baseLearners[ib] = tmpPair.second.first.first;				
			delete tmpBL;				

			tmpPairPos.second.first.second.first = ib;
			tmpPairNeg.second.first.second.first = ib;
						
			if ( ib > 0 ) {
				//set the descendant idx
				_idxPairs[ parentIdx ][ tmpPair.second.first.second.second ] = ib;
			}

			ib++;

			if ( ib >= _numBaseLearners ) break;


			
			//extend positive node
			if ( tmpPairPos.second.first.first ) {
				bLearnerVector = calculateChildrenAndEnergies( tmpPairPos );
			} else {
				bLearnerVector.clear();
			}

			// if the energie is getting higher then we push it into the priority queue
			if ( ! bLearnerVector.empty() ) 
			{
				if (_verbose > 2) {
					cout << "Edges: (parent, pos, neg): " << bLearnerVector[0].first << " " << bLearnerVector[1].first << " " << bLearnerVector[2].first << endl << flush;
					//cout << "alpha[" << (ib) <<  "] = " << _alpha << endl << flush;
				}

				// if the energy is getting higher then we push it into the priority queue
				if ( bLearnerVector[0].first < ( bLearnerVector[1].first + bLearnerVector[2].first ) ) {
					float deltaEdge = abs( 	bLearnerVector[0].first - ( bLearnerVector[1].first + bLearnerVector[2].first ) );
					innerNode.first = deltaEdge;
					innerNode.second = bLearnerVector;
					pq.push( innerNode ); 
				} else {
					//delete bLearnerVector[0].second.first.first;
					delete bLearnerVector[1].second.first.first;
					delete bLearnerVector[2].second.first.first;
				
					if ( ib >= _numBaseLearners ) {
						delete tmpPairPos.second.first.first;
						break;
					} else {
						//this will be a leaf, we do not extend further
						int parentIdx = tmpPairPos.second.first.second.first;
						BaseLearner* tmpBL = _baseLearners[ib];
						_baseLearners[ib] = tmpPairPos.second.first.first;				
						delete tmpBL;
						_idxPairs[ parentIdx ][ tmpPairPos.second.first.second.second ] = ib;
						ib += 1;
					}
				}
			} 



			//extend negative node
			if ( tmpPairNeg.second.first.first ) {
			bLearnerVector = calculateChildrenAndEnergies( tmpPairNeg );
			} else {
				bLearnerVector.clear();
			}

			// if the energie is getting higher then we push it into the priority queue
			if ( ! bLearnerVector.empty() ) 
			{
				if (_verbose > 2) {
					cout << "Edges: (parent, pos, neg): " << bLearnerVector[0].first << " " << bLearnerVector[1].first << " " << bLearnerVector[2].first << endl << flush;
					//cout << "alpha[" << (ib) <<  "] = " << _alpha << endl << flush;
				}

				// if the energie is getting higher then we push it into the priority queue
				if ( bLearnerVector[0].first < ( bLearnerVector[1].first + bLearnerVector[2].first ) ) {
					float deltaEdge = abs( 	bLearnerVector[0].first - ( bLearnerVector[1].first + bLearnerVector[2].first ) );
					innerNode.first = deltaEdge;
					innerNode.second = bLearnerVector;
					pq.push( innerNode ); 
				} else {
					//delete bLearnerVector[0].second.first.first;
					delete bLearnerVector[1].second.first.first;
					delete bLearnerVector[2].second.first.first;

					if ( ib >= _numBaseLearners ) {
						delete tmpPairNeg.second.first.first;
						break;
					} else {
						//this will be a leaf, we do not extend further
						int parentIdx = tmpPairNeg.second.first.second.first;
						BaseLearner* tmpBL = _baseLearners[ib];
						_baseLearners[ib] = tmpPairNeg.second.first.first;				
						delete tmpBL;
						_idxPairs[ parentIdx ][ tmpPairNeg.second.first.second.second ] = ib;
						ib += 1;
					}


				}
			}

		}		
		

		for(int ib2 = ib; ib2 < _numBaseLearners; ++ib2) delete _baseLearners[ib2];
		_numBaseLearners = ib;

		if (_verbose > 2) {
			cout << "Num of learners: " << _numBaseLearners << endl << flush;
		}

		//clear the priority queur 
		while ( ! pq.empty() && ( ib < _numBaseLearners ) ) {
			//get the best learner from the priority queue
			innerNode = pq.top();
			pq.pop();
			bLearnerVector = innerNode.second;

			delete bLearnerVector[0].second.first.first;
			delete bLearnerVector[1].second.first.first;
			delete bLearnerVector[2].second.first.first;
		}

		_id = _baseLearners[0]->getId();
		for(int ib = 0; ib < _numBaseLearners; ++ib)
			_id += "_x_" + _baseLearners[ib]->getId();

		//calculate alpha
		this->_alpha = 0.0;
		float eps_min = 0.0, eps_pls = 0.0;

		_pTrainingData->clearIndexSet();
		for( int i = 0; i < _pTrainingData->getNumExamples(); i++ ) {
			vector< Label> l = _pTrainingData->getLabels( i );
			for( vector< Label >::iterator it = l.begin(); it != l.end(); it++ ) {
				float result  = this->classify( _pTrainingData, i, it->idx );

				if ( ( result * it->y ) < 0 ) eps_min += it->weight;
				if ( ( result * it->y ) > 0 ) eps_pls += it->weight;
			}

		}

		this->_alpha = getAlpha( eps_min, eps_pls );

		// calculate the energy (sum of the energy of the leaves
		float energy = this->getEnergy( eps_min, eps_pls );

		return energy;
}
// -----------------------------------------------------------------------

vector<floatBaseLearner> TreeLearner2::calculateChildrenAndEnergies( floatBaseLearner& bLearner ) {
	vector<floatBaseLearner> retval;
	floatBaseLearner tmpPair;
	retval.push_back( bLearner );

	_pTrainingData->loadIndexSet( bLearner.second.second );

	//separate the dataset
	set< int > idxPos, idxNeg;
	idxPos.clear();
	idxNeg.clear();
	float phix;

	for (int i = 0; i < _pTrainingData->getNumExamples(); ++i) {
		// this returns the phi value of classifier
		phix = bLearner.second.first.first->classify(_pTrainingData,i,0);
		if ( phix <  0 )
			idxNeg.insert( _pTrainingData->getRawIndex( i ) );
		else if ( phix > 0 ) { // have to redo the multiplications, haven't been tested
			idxPos.insert( _pTrainingData->getRawIndex( i ) );
		}
	}

	if ( (idxPos.size() < 1 ) || (idxNeg.size() < 1 ) ) {
		retval.clear();
		return retval;
	}
	
	_pTrainingData->loadIndexSet( idxPos );
	
	if ( ! _pTrainingData->isSamplesFromOneClass() ) {
		BaseLearner* posLearner = _baseLearners[0]->copyState();

		posLearner->run();
		float posEdge = getEdge( posLearner, _pTrainingData );

		tmpPair.first = posEdge;
		tmpPair.second.first.first = posLearner;
		//set the parent idx to zero
		tmpPair.second.first.second.first = 0;
		//this means that it will be a left child in the tree
		tmpPair.second.first.second.second = 0;
		tmpPair.second.second = idxPos;
	} else {
		BaseLearner* pConstantWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner("ConstantLearner");

		BaseLearner* posLearner = pConstantWeakHypothesisSource->create();
		posLearner->setTrainingData(_pTrainingData);
		float constantEnergy = posLearner->run();


		//BaseLearner* posLearner = _baseLearners[0]->copyState();
		float posEdge = getEdge( posLearner, _pTrainingData );

		tmpPair.first = posEdge;
		tmpPair.second.first.first = posLearner;
		//set the parent idx to zero
		tmpPair.second.first.second.first = 0;
		//this means that it will be a left child in the tree
		tmpPair.second.first.second.second = 0;
		tmpPair.second.second = idxPos;
	}

	retval.push_back( tmpPair );

	_pTrainingData->loadIndexSet( idxNeg );

	if ( ! _pTrainingData->isSamplesFromOneClass() ) {
		BaseLearner* negLearner = _baseLearners[0]->copyState();

		
		negLearner->run();
		float negEdge = getEdge( negLearner, _pTrainingData );

		tmpPair.first = negEdge;
		tmpPair.second.first.first = negLearner;
		//set the parent idx to zero
		tmpPair.second.first.second.first = 0;
		//this means that it will be a right child in the tree
		tmpPair.second.first.second.second = 1;
		tmpPair.second.second = idxNeg;
	} else {
		BaseLearner* pConstantWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner("ConstantLearner");

		BaseLearner* negLearner =  pConstantWeakHypothesisSource->create();
		negLearner->setTrainingData(_pTrainingData);
		float constantEnergy = negLearner->run();


		tmpPair.first = getEdge( negLearner, _pTrainingData );;
		tmpPair.second.first.first = negLearner;
		//set the parent idx to zero
		tmpPair.second.first.second.first = 0;
		//this means that it will be a right child in the tree
		tmpPair.second.first.second.second = 1;
		tmpPair.second.second = idxNeg;
	}

	retval.push_back( tmpPair );

	return retval;
}


// -----------------------------------------------------------------------

void TreeLearner2::save(ofstream& outputStream, int numTabs)
{
	// Calling the super-class method
	BaseLearner::save(outputStream, numTabs);

	// save numBaseLearners
	outputStream << Serialization::standardTag("numBaseLearners", _numBaseLearners, numTabs) << endl;

	for( int ib = 0; ib < _numBaseLearners; ++ib ) {
		outputStream << Serialization::standardTag("leftChild", _idxPairs[ib][0], numTabs) << endl;
		outputStream << Serialization::standardTag("rightChild", _idxPairs[ib][1], numTabs) << endl;
	}

	for( int ib = 0; ib < _numBaseLearners; ++ib ) {
		_baseLearners[ib]->save(outputStream, numTabs + 1);
	}
}

// -----------------------------------------------------------------------

void TreeLearner2::load(nor_utils::StreamTokenizer& st)
{
	BaseLearner::load(st);

	_numBaseLearners = UnSerialization::seekAndParseEnclosedValue<int>(st, "numBaseLearners");
	//   _numBaseLearners = 2;
	_idxPairs.clear();
	for(int ib = 0; ib < _numBaseLearners; ++ib) {
		int leftChild = UnSerialization::seekAndParseEnclosedValue<int>(st, "leftChild");
		int rightChild = UnSerialization::seekAndParseEnclosedValue<int>(st, "rightChild");
		vector< int > p( 2, -1 );
		p[0] = leftChild;
		p[1] = rightChild;
		_idxPairs.push_back( p );
	}


	for(int ib = 0; ib < _numBaseLearners; ++ib) {
		UnSerialization::loadHypothesis(st, _baseLearners, _pTrainingData, _verbose);
	}

}

// -----------------------------------------------------------------------

void TreeLearner2::subCopyState(BaseLearner *pBaseLearner)
{
	BaseLearner::subCopyState(pBaseLearner);

	TreeLearner2* pTreeLearner2 =
		dynamic_cast<TreeLearner2*>(pBaseLearner);

	pTreeLearner2->_numBaseLearners = _numBaseLearners;

	// deep copy
	for(int ib = 0; ib < _numBaseLearners; ++ib)
		pTreeLearner2->_baseLearners.push_back(_baseLearners[ib]->copyState());
}

// -----------------------------------------------------------------------

} // end of namespace shogun
