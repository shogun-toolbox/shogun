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

// classhierarchy.cpp : Defines the entry point for the console application.
//

#include "ClassHierarchy.h"
#include <stdlib.h>

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

//--------------------------------------------------------------------

ClassHierarchy::~ClassHierarchy(void)
{
	for( int i=0; i < _root.getNumOfChildren(); i++ ) {
		InnerNode* currNode = _root.getithChild( i );
		eraseHierarchy( currNode );
		delete currNode;
	}
}

//--------------------------------------------------------------------

void ClassHierarchy::eraseHierarchy( InnerNode* currNode ) {
	for( int i=0; i < currNode->getNumOfChildren(); i++ ) {
		InnerNode* childNode = currNode->getithChild( i );
		if ( childNode->isLeaf() ) delete childNode;
		else eraseHierarchy( childNode );	
	}
}

//--------------------------------------------------------------------

void ClassHierarchy::subcat( int category ) {
	
	InnerNode* currNode;
	InnerNode* siblingNode;

	vector<int> ancestorCategories;
	vector<int> siblingCategories;

	getAncestors( ancestorCategories, category );
	currNode = &_root;
	
	//erase all nodes except the node having the input category and its descendants
	for( int i=0; i < ancestorCategories.size(); i++ ) {
		currNode = convertCategroyToInnerNode( ancestorCategories[i] );
		getSiblings( siblingCategories, currNode->getCategory() );
		for( int j=0; j < siblingCategories.size(); j++ ) {
			siblingNode = convertCategroyToInnerNode( siblingCategories[j] );
			if ( siblingNode->getCategory() != currNode->getCategory() ) {
				eraseHierarchy( siblingNode );
				delete siblingNode;
				siblingNode = NULL;
			}
		}
		//we don't delete the new root	
		if ( i < ancestorCategories.size()-1 ) {
			delete currNode;
			currNode = NULL;
		}
	}
	
	_root.clear();
	//copy the children into the root element, and update their parent pointer
	_root.adoptChildren( currNode );
	delete currNode;

	updateMemberVariables();
}

//--------------------------------------------------------------------

void ClassHierarchy::eraseTreeUptoDepth( int depth )  {
	for( int i=0; i<_root.getNumOfChildren(); i++ ) {
		InnerNode* currNode = _root.getithChild( i );
		eraseTreeUptoDepthPostFix( currNode, 1, depth );
	}
	updateMemberVariables();
}

//--------------------------------------------------------------------
void ClassHierarchy::getCategorySet( set<int>& categories ){
	vector<int> tmpVec;
	vector<int>::iterator it;
	
	categories.clear();

	for( int i=0; i<_root.getNumOfChildren(); i++ ) {
		InnerNode* currNode = _root.getithChild( i );
		getDescendants( tmpVec, currNode->getCategory() );

		for( it = tmpVec.begin(); it != tmpVec.end(); it++ ) {
			categories.insert( *it );
		}
	}
}

//--------------------------------------------------------------------

void ClassHierarchy::keepTheChildrenOfACategory( int category ) {
	//a memoria leakekt ki kell majd kuszobolni
	InnerNode* currNode = convertCategroyToInnerNode( category );
	
	_root.adoptChildren( currNode );
	eraseTreeUptoDepth( 1 );
	
	updateMemberVariables();
}

//--------------------------------------------------------------------

void ClassHierarchy::eraseTreeUptoDepthPostFix( InnerNode* currNode, int currentdepth, int depth )  {
	for( int i=0; i< currNode->getNumOfChildren(); i++ ) {
		InnerNode* childNode = currNode->getithChild( i );
		eraseTreeUptoDepthPostFix( childNode, currentdepth+1, depth );
	}

	if (  currentdepth >= depth ) currNode->eraseChildren(); 
}


//--------------------------------------------------------------------

void ClassHierarchy::updateMemberVariables( void ){
	
	_mapIdxToCategory.clear();
	_mapCategoryToIdx.clear();
	_mapCategoryToNode.clear();
	_classMap.clear();

	for( int i=0; i < _root.getNumOfChildren(); i++ ) {
		InnerNode* currNode = _root.getithChild( i );
		collectCategories( currNode, _classMap, _mapIdxToCategory, _mapCategoryToIdx, _mapCategoryToNode );
	}
	 
	_numOfCategories = _mapCategoryToIdx.size();
}

//--------------------------------------------------------------------

void ClassHierarchy::collectCategories( InnerNode* currNode, NameMap& namemap, map<int,int>& idxtocat, map<int,int>& cattoidx, map<int,InnerNode*>& cattonode ) {
	int curridx = idxtocat.size();

	idxtocat.insert( make_pair<int,int>( curridx, currNode->getCategory() ) );
	cattoidx.insert( make_pair<int,int>( currNode->getCategory(), curridx ) );
	cattonode.insert( make_pair<int,InnerNode*>( currNode->getCategory(), currNode ) );

	string s = nor_utils::int2string( currNode->getCategory() );
	namemap.addName( s );

	for( int i=0; i < currNode->getNumOfChildren(); i++ ) {
		InnerNode* childNode = currNode->getithChild( i );
		collectCategories( childNode, namemap, idxtocat, cattoidx, cattonode );
	}
	
}

//--------------------------------------------------------------------

void ClassHierarchy::load( const string fname )
{
	ifstream infile( fname.c_str() );

	if ( ! infile.is_open() ) {
		cout << "Hiearachy file doesn't exist" << endl;
		exit( -1 );
	}

	string line;
	istringstream ss;
	ss.imbue( _locale );
	int tmpCat;
	vector<int> hiearchyPath(0);
	
	
	while( ! infile.eof() ) {
		getline( infile, line );

		if ( line.length() == 0 ) { 
			//empty line
			continue;
		}

		line = nor_utils::trim( line );
		
		//cout << line << endl;
		ss.clear();
		ss.str( line );

		hiearchyPath.clear();

		while (!ss.eof())
		{
			ss >> tmpCat;
			//cout << tmpCat << " ";
			hiearchyPath.push_back( tmpCat );
		}

		//cout << endl;
		// hiearachPath contains the category names
		addHierarchPath( hiearchyPath );
		
		//while ( infile.get() != '\n' && !infile.eof() );
	}


	infile.close();
}

//--------------------------------------------------------------------

void ClassHierarchy::addHierarchPath( vector<int>& hierarchyPath ){
	InnerNode* currNode = &_root;
	for( vector<int>::iterator it = hierarchyPath.begin(); it != hierarchyPath.end(); it++ ) {
		if ( currNode->hasChildWithThisCategory( *it ) ) {
			currNode = currNode->getChild( *it );
		} else {
			if ( existCategory( *it ) ) {
				cout << "This category" << *it <<  " exist anywhere in the hierarchy!!!!" << endl;
				exit( -1 );
			}
			addChild( currNode, *it );
			
			_mapIdxToCategory.insert( make_pair<int,int>( _numOfCategories, *it ) );
			_mapCategoryToIdx.insert( make_pair<int,int>( *it, _numOfCategories ) );
			_numOfCategories++;

			currNode = currNode->getChild( *it );
			_mapCategoryToNode.insert( make_pair<int,InnerNode*>( *it, currNode ) );

			string s;
			stringstream ss;
			ss << *it;
			s = ss.str();

			_classMap.addName( s );
		}
	}
}

//--------------------------------------------------------------------

void ClassHierarchy::addChild( InnerNode* parent, int category ){
	InnerNode* child = new InnerNode( parent, category );
	parent->addChild( child );
}

}


