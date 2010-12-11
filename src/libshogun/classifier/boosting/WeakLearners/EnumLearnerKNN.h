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
* \file SingleStumpLearner.h A single threshold decision stump learner. 
*/

#ifndef __ENUM_LEARNER_KNN_H
#define __ENUM_LEARNER_KNN_H

#include "FeaturewiseLearner.h"
#include "classifier/boosting/Utils/Args.h"

#include <vector>
#include <fstream>
#include <cassert>
#include <queue>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {

class KNNGraph;
/**
* A \b single threshold decision stump learner. 
* There is ONE and ONE ONLY threshold here.
*/
class EnumLearnerKNN : public FeaturewiseLearner
{
public:
   /**
   * Declare weak-learner-specific arguments.
   * adding --baselearnertype
   * \param args The Args class reference which can be used to declare
   * additional arguments.
   * \date 24/04/2007
   */
   virtual void declareArguments(nor_utils::Args& args);

   /**
   * Set the arguments of the algorithm using the standard interface
   * of the arguments. Call this to set the arguments asked by the user.
   * \param args The arguments defined by the user in the command line.
   * \date 24/04/2007
   */
   virtual void initLearningOptions(const nor_utils::Args& args);

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~EnumLearnerKNN() {}

   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \date 21/05/2007
   */
   virtual BaseLearner* subCreate() { return new EnumLearnerKNN(); }

   /**
   * Run the learner to build the classifier on the given data.
   * \param pData The pointer to the data.
   * \see BaseLearner::run
   * \date 21/05/2007
   */
   virtual float run();
   virtual float run( int colIdx );
   /**
   * Save the current object information needed for classification,
   * that is the _u vector.
   * \param outputStream The stream where the data will be saved
   * \param numTabs The number of tabs before the tag. Useful for indentation
   * \remark To fully save the object it is \b very \b important to call
   * also the super-class method.
   * \date 21/05/2007
   */
   virtual void save(ofstream& outputStream, int numTabs = 0);

   /**
   * Load the xml file that contains the serialized information
   * needed for the classification and that belongs to this class.
   * \param st The stream tokenizer that returns tags and values as tokens
   * \see save()
   * \date 21/05/2007
   */
   virtual void load(nor_utils::StreamTokenizer& st);

   /**
   * Copy all the info we need in classify().
   * pBaseLearner was created by subCreate so it has the correct (sub) type.
   * Usually one must copy the same fields that are loaded and saved. Don't 
   * forget to call the parent's subCopyState().
   * \param pBaseLearner The sub type pointer into which we copy.
   * \see save
   * \see load
   * \see classify
   * \see ProductLearner::run()
   * \date 25/05/2007
   */
   virtual void subCopyState(BaseLearner *pBaseLearner);

   /**
   * The same discriminative function as below, but called with a data point. 
   * \param pData The input data.
   * \param pointIdx The index of the data point.
   * \return \phi[(int)pointIdx]
   * \date 21/05/2007
   */
   virtual float phi(float val, int classIdx) const;



protected:

   vector<float> _u;
   float		 _uOffset;
   static KNNGraph _kNN;
};

//////////////////////////////////////////////////////////////////////////
typedef map< int, int > Neighborhood; 
typedef vector< Neighborhood > GraphRepresentation;

class KNNGraph {
public:
	KNNGraph() {
		_isReady = false;
		_pTrainingData = NULL;
		_FeatureWiseKNNGraph.clear();
		_k = 5;
		_fname = "knngraph.txt";
	}

	virtual string getName() { return _fname; }
	virtual void setName( string fname ) { _fname = fname; }

	virtual void calculagteKNNGraph() {
		if ( ! _isReady ) {
			if ( ! load() ) {
				calculateFeatureWiseKNNGraph( 0, 1 );
				calculateFeatureWiseKNNGraph( 1, 0 );
				save();
			} 
			_isReady = true;
		}
	}

	virtual void setTrainingData(InputData *pTrainingData) {
		_pTrainingData = pTrainingData;

		// allocate memory for KNN graph
		_FeatureWiseKNNGraph.resize( _pTrainingData->getNumAttributes() );
		for( int i = 0; i < _pTrainingData->getNumAttributes(); i++ ) {
			_FeatureWiseKNNGraph[i].resize( _pTrainingData->getNumExamples() );
		}
	}
	
	virtual bool isReady() { return _isReady; }
	
	virtual void getjthFeatureithExampleNeighborhood( int i, int j, vector< int >& neighbors ) {
		neighbors.resize( _FeatureWiseKNNGraph[i][j].size() );
		map< int, int >::iterator it;
		int ib;

		for( ib = 0, it = _FeatureWiseKNNGraph[i][j].begin(); it != _FeatureWiseKNNGraph[i][j].end(); it++, ib++  ) {
			neighbors[ib] = it->first;
		}
	}

	virtual void setK( int k ) { _k = k; }
	virtual int getK() { return _k; }

	virtual void save() {
		ofstream outFile;
		outFile.open( _fname.c_str() );
	    
		if (!outFile.is_open())
	    {
		   cerr << "ERROR: Cannot open kNN file <" << _fname << ">!" << endl;
		   exit(1);
	    }
		
		for( int i1 = 0; i1 < _FeatureWiseKNNGraph.size(); i1 ++ ) {
			for( int i2 = 0; i2 < _FeatureWiseKNNGraph[i1].size(); i2++ ) {
				for( map<int,int>::iterator it = _FeatureWiseKNNGraph[i1][i2].begin(); it != _FeatureWiseKNNGraph[i1][i2].end(); it++ ) {
					outFile << i1 << " " << i2 << " " << it->first << " " << it->second << endl; 
				}
			}
		}

		outFile.close();

	}

	virtual bool load() {
		if ( ! isExistKNNFile() ) {
			return false;
		}
		   ifstream inFile( _fname.c_str());
		if (!inFile.is_open())
		{
			cerr << "ERROR: Cannot open knn file <" << _fname << ">!" << endl;
			exit(1);
		}
		
	    nor_utils::StreamTokenizer st(inFile, " \n\r\t");
		string token;
		int i1, i2, i3, i4;

		while ( st.has_token() ) {
			token = st.next_token();
			i1 = atoi( token.c_str() );
			if ( ! st.has_token() ) break;

			token = st.next_token();
			i2 = atoi( token.c_str() );

			token = st.next_token();
			i3 = atoi( token.c_str() );
			
			token = st.next_token();
			i4 = atoi( token.c_str() );
			

			//cout << i1 << " " << i2 << " " << i3 << " " << i4 << endl; 
			_FeatureWiseKNNGraph[i1][i2].insert( map<int, int>::value_type( i3, i4 ) );
		}

		inFile.close();

		return true;
	}

	virtual bool isExistKNNFile() {
		if( ifstream( _fname.c_str() ) ) {
			//cout << "exists!";
			return true;
		} else return false;
	}

protected:
	virtual void calculateFeatureWiseKNNGraph( int sortedAttribute, int targetAttribute ) {
		const int numClasses = _pTrainingData->getNumClasses();
		const int numAttribute = _pTrainingData->getNumAttributes();
		const int numExamples = _pTrainingData->getNumExamples();

		//const int targetObjectNum = _pTrainingData->getEnumMap( attributeNum ).getNumNames();
		//const int objectNum = _pTrainingData->getEnumMap( 1 ).getNumNames();


		const int userNum = _pTrainingData->getEnumMap( sortedAttribute ).getNumNames();
		const int objectNum = _pTrainingData->getEnumMap( targetAttribute ).getNumNames();

		cout << "Allocate memory for vote matrix...";

		vector< map< int, float> > voteMatrix( userNum );
						
		//_clusters.resize( userNum );

		priority_queue< pair< float, int > > simMatrix;
		
		cout << "Ready!" << endl;

		for( int i = 0; i<numExamples; i++ ) 
		{
			int userId =  (int) _pTrainingData->getValue( i, sortedAttribute  );
			int objectId = (int) _pTrainingData->getValue( i, targetAttribute );
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
		
		for( int i = 0; i < userNum; i++ )  {
			while( ! simMatrix.empty() ) {
				simMatrix.pop();
			}

			for( int j = 0; j<userNum; j++ ) {
				if ( i == j ) continue;

				float similarity = 0.0;
				int voteNum = 0;
				for( map<int,float>::iterator it1 = voteMatrix[i].begin(); it1 != voteMatrix[i].end(); it1++ ) {
					
					map< int, float >::iterator it2 = voteMatrix[j].find( it1->first );
					// by the calculation of diffrence of votes we also take into account the value of rating
					if ( it2 != voteMatrix[j].end() ) {
						similarity += ( ( it1->second - it2->second ) * ( it1->second - it2->second ) );
						voteNum++;
					}
				}

				if ( voteNum > 0 ) { 
					similarity /= (float)voteNum;
					similarity = -similarity;

					pair< float, int > tmpPair1( similarity, j );
					simMatrix.push( tmpPair1 );
				}
			}

			size_t st = _k;
			if ( simMatrix.size() < _k ) st = simMatrix.size();

			//_clusters[i].resize( st );
			int j = 0;
			float sum = 0.0;
			while ( ( ! simMatrix.empty() ) && ( j < _k ) ) {
				pair< float, int > tmpPair = simMatrix.top();
				simMatrix.pop();

				//_clusters[i][j] = tmpPair.second;
				_FeatureWiseKNNGraph[sortedAttribute][ i ].insert( map<int,int>::value_type( tmpPair.second,  j ) );
				j++;
			}

		}		

    	//write out the similarity matrix
		
		ofstream out;
		out.open( "sim.txt" );
		
		for( int i = 0; i<userNum; i++ )  {
			for( map< int, int >::iterator it = _FeatureWiseKNNGraph[sortedAttribute][i].begin(); it != _FeatureWiseKNNGraph[sortedAttribute][i].end(); it++ ) {
				out << it->first << " " << it->second << " ";
			}
			out << endl;
			out.flush();
		}
		
		out.close();

		out.clear();

		
		/*
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
		cout << "Similarity matrix ready..." << endl;
	}
	
	bool _isReady;
	InputData* _pTrainingData;
	vector< GraphRepresentation > _FeatureWiseKNNGraph;	
	int _k;
	string _fname;
};



} // end of namespace shogun

#endif // __ENUM_LEARNER_H
