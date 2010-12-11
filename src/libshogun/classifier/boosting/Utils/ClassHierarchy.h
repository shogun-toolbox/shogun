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
* \file BaseLearner.h The abstract basic (weak) learner.
*/
#pragma warning( disable : 4786 )

#ifndef __CLASS_HIERARCHIES_H
#define __CLASS_HIERARCHIES_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>

#include "classifier/boosting/IO/NameMap.h"
#include "classifier/boosting/Utils/Utils.h"

using namespace std;


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {


class InnerNode {
public:
	InnerNode( void ) : _category(-1), _parent( NULL ), _childs(0) {
	}

	InnerNode( InnerNode* parent, int category ) {
		InnerNode();
		_parent = parent;
		_category = category;
	}

	bool isRoot() { return _parent == NULL; }

	InnerNode* getParent( void ) { return _parent; } 
	void setParent( InnerNode* parent ) { _parent = parent; } 

	bool hasChildWithThisCategory( int category ) {
		map<int,size_t>::iterator it = _childCategories.find( category );
		return ( ! ( it == _childCategories.end() ) );
	}

	void addChild( InnerNode* node ) {
		_childs.push_back( node );	
		_childCategories.insert( make_pair<int,size_t>(node->getCategory(), _childs.size()-1) );
	}

	InnerNode* getChild( int category ) {
		map<int,size_t>::iterator it = _childCategories.find( category );
		return _childs[ it->second ];
	}

	InnerNode* getithChild( int idx ) {
		return _childs[ idx ];
	}


	void getChildCategories( vector<int>& childCategories ) {
		childCategories.clear();
		for( vector<InnerNode*>::iterator it = _childs.begin(); it != _childs.end(); it++ ) {
			childCategories.push_back( (*it)->getCategory() );
		}
	}

	bool isLeaf() { return _childCategories.size() == 0; }
	int getCategory( void ) { return _category; };
	int getNumOfChildren() { return _childs.size(); }
	
	void clear( void ) {
		_parent = NULL;
		_childs.clear();
		_childCategories.clear();
		_category = -1;
	}

	void adoptChildren( InnerNode* node ) {
		_childs.clear();
		_childs.resize( node->getNumOfChildren() );

		for( int i=0; i<node->getNumOfChildren(); i++ ) {
			_childs[i] = node->getithChild( i );
			_childs[i]->setParent( this );
		}
	}

	
	void eraseChildren( void ) {
		for( vector<InnerNode*>::iterator it = _childs.begin(); it != _childs.end(); it++ ) {
			delete *it;
		}
		_childCategories.clear();
		_childs.clear();
	}

protected:
	InnerNode*			_parent;
	vector<InnerNode*>	_childs;
	//contains the category and the idx pair concerning the idx in the childs vector
	map<int,size_t>		_childCategories;
	// the category of the node
	int					_category;
};

class ClassHierarchy
{
public:
	ClassHierarchy(void) : _fname( "cat_hier.txt" ), _numOfCategories(0) {
		//_locale = locale(locale(), new nor_utils::white_spaces(", "));
	}

	~ClassHierarchy(void);
	// erase a subtree, but it doesn't delete currNode
	void eraseHierarchy( InnerNode* currNode );
	void load( const string fname );
	
	
	void getAncestors( vector<int>& ancestorCategories, int category ) {
		if ( ! existCategory( category ) ) {
			ancestorCategories.clear();
			return;
		}

		vector<int> tmpVec(0);
		
		map<int,InnerNode*>::iterator it = _mapCategoryToNode.find( category );
		InnerNode* currNode = it->second;

		while ( ! currNode->isRoot() ) {
			tmpVec.push_back( currNode->getCategory() );
			currNode = currNode->getParent();
		}
		ancestorCategories.resize( tmpVec.size() );
		copy( tmpVec.rbegin(), tmpVec.rend(), ancestorCategories.begin() );
	}

	void getSiblings( vector<int>& siblings, int category ) {
		if ( ! existCategory( category ) ) {
			siblings.clear();
			return;
		}

		map<int,InnerNode*>::iterator it = _mapCategoryToNode.find( category );
		InnerNode* currNode = it->second;

		//go to its parent
		currNode = currNode->getParent();
		currNode->getChildCategories( siblings );	
	}

	void getDescendants( vector<int>& descendants, int category ) {
		if ( ! existCategory( category ) ) {
			descendants.clear();
			return;
		}

		map<int,InnerNode*>::iterator it = _mapCategoryToNode.find( category );
		InnerNode* currNode = it->second;
		vector<int> children;

		descendants.clear();
		descendants.push_back( currNode->getCategory() );
		
		size_t currIdx = 0;

		while ( currIdx < descendants.size() ) {
			currNode = convertCategroyToInnerNode( descendants[ currIdx++ ] );
			currNode->getChildCategories( children );
			
			for( vector<int>::iterator it = children.begin(); it != children.end(); it++ ) descendants.push_back( *it );
		}
	}

	int getParent( int category );

	int convertIdxToCategory( int category ) { 
		map<int,int>::iterator it = _mapIdxToCategory.find( category );
		if ( it == _mapIdxToCategory.end() ) {
			return -1;
		} else {
			return it->second;
		}
	}

	int convertCategoryToIdx( int idx ) { 
		map<int,int>::iterator it = _mapCategoryToIdx.find( idx );
		if ( it == _mapCategoryToIdx.end() ) {
			return -1;
		} else {
			return it->second;
		}
	}


	bool existCategory( int category ) {
		map<int,InnerNode*>::iterator it = _mapCategoryToNode.find( category );
		return ( ! (it == _mapCategoryToNode.end() ) );
	}

	void getClassNameMap( NameMap& classNameMap ) { classNameMap = _classMap; }

	////////////////////////////////////////////////////////////////////////////
	// ide kerulnek a megszoritasok a kategoriakban
	///////////////////////////////////////////////////////////////////////////
	// csak egy belso kategoriat reszfajat tartjuk  //LS_SUBCAT
	void subcat( int category );
	// egy bizonyos melysegik hasznaljuk a kategoriakat //LS_DEPTH
	void eraseTreeUptoDepth( int depth );
protected:
	void eraseTreeUptoDepthPostFix( InnerNode* currNode, int currentdepth, int depth );
public:
	void keepTheChildrenOfACategory( int category );
	///////////////////////////////////////////////////////////////////////////

	InnerNode* convertCategroyToInnerNode( int category ) {
		map<int,InnerNode*>::iterator it = _mapCategoryToNode.find( category );
		return it->second;
	}

	void getCategorySet( set<int>& categories );

	int getNumOfCategories( void ) { return _numOfCategories; }
protected:
	void updateMemberVariables( void );
	void collectCategories( InnerNode* currNode, NameMap& namemap, map<int,int>& idxtocat, map<int,int>& cattoidx, map<int,InnerNode*>& cattonode );

	void addHierarchPath( vector<int>& hierarchyPath );
	void addChild( InnerNode* parent, int category );

	string				_fname;

	int					_numOfCategories;
	map<int,int>		_mapIdxToCategory;
	map<int,int>		_mapCategoryToIdx;
	map<int,InnerNode*>	_mapCategoryToNode;

	InnerNode			_root;
	NameMap				_classMap;
	locale				_locale;
};


}

#endif

