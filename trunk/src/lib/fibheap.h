/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _FIBHEAP_H
#define _FIBHEAP_H

//***************************************************************************
// The Fibonacci heap implementation contained in FIBHEAP.H and FIBHEAP.CPP
// is Copyright (c) 1996 by John Boyer
//
// Once this Fibonacci heap implementation (the software) has been published
// by Dr. Dobb's Journal, permission to use and distribute the software is
// granted provided that this copyright notice remains in the source and
// and the author (John Boyer) is acknowledged in works that use this program.
//
// Every effort has been made to ensure that this implementation is free of
// errors.  Nonetheless, the author (John Boyer) assumes no liability regarding
// your use of this software.
//
// The author would also be very glad to hear from anyone who uses the
// software or has any feedback about it.
// Email: jboyer@gulf.csc.uvic.ca
//***************************************************************************

/******************
*
*  Class HeapNode added by an unknown author - it allows you to store a double and a long int, 
*    with sorting done on the former, i.e. the key is the double.
*
*  Class Heap added by dinoj@uchicago.edu 13/07/05
*  
*  Example usage
*
*  Heap h;
*  h.Insert(1.123,  10);                                // now the heap has 1.123 (and associated 10)
*  h.Insert(-1.123,  20);                               // now the heap has 1.123, -1.123 (and associated 10, 20)
*  long int a = 100;                                     
*  double d = 43.12;
*  h.Insert(d,a);                                       // now the heap has 1.123, -1.123, 43.12 (and associated 10, 20, 100)
*  h.Minimum(d,a);                                      // this places the smallest key in d and its associated field in a. 
*                                                          They are not deleted from the heap, which still has 3 elements.
*  cout << d << " " << a << h.GetNumNodes() << "\n"     // this should output  "-1.123   20   3" 
*  h.ExtractMin(d,a);                                   // Same as before, but this time they are deleted, so the heap has two elements (1.123, 43.12) 
*  cout << d << " " << a << h.GetNumNodes() << "\n"     // this should output  "-1.123   20   2" 
*  h.ExtractMin(d,a);                                   
*  cout << d << " " << a << h.GetNumNodes() << "\n"     // this should output  " 1.123   10   1"       // one element left in the heap!
*
* Note: There is no safety checking in ExtractMin to see if there are any elements in the heap to extract. 
*       If you want that, use ExtractMin_safe .
****************************************************************************************/

#define OK      0
#define NOTOK   -1

//======================================================
// Fibonacci Heap Node Class
//======================================================

class FibHeap;

class FibHeapNode
{
friend class FibHeap;

     FibHeapNode *Left, *Right, *Parent, *Child;
     short Degree, Mark, NegInfinityFlag;

protected:

     inline int  FHN_Cmp(FibHeapNode& RHS);
     inline void FHN_Assign(FibHeapNode& RHS);

public:

     FibHeapNode();                // worth inlining
     virtual ~FibHeapNode();

     virtual void operator =(FibHeapNode& RHS);
     virtual int  operator ==(FibHeapNode& RHS);
     virtual int  operator <(FibHeapNode& RHS);

     //     virtual void Print();
};

//========================================================================
// Fibonacci Heap Class
//========================================================================

class FibHeap
{
     FibHeapNode *MinRoot;
     long NumNodes, NumTrees, NumMarkedNodes;

     int HeapOwnershipFlag;

public:

     FibHeap();
     virtual ~FibHeap();

// The Standard Heap Operations

     void Insert(FibHeapNode *NewNode);
     void Union(FibHeap *OtherHeap);

     FibHeapNode *Minimum();    // worth inlining

     FibHeapNode *ExtractMin();

     int DecreaseKey(FibHeapNode *theNode, FibHeapNode& NewKey);
     int Delete(FibHeapNode *theNode);

// Extra utility functions

     int  GetHeapOwnership() { return HeapOwnershipFlag; };
     void SetHeapOwnership() { HeapOwnershipFlag = 1; };
     void ClearHeapOwnership() { HeapOwnershipFlag = 0; };

     long GetNumNodes() { return NumNodes; };
     long GetNumTrees() { return NumTrees; };
     long GetNumMarkedNodes() { return NumMarkedNodes; };

     //     void Print(FibHeapNode *Tree = 0, FibHeapNode *theParent=0);

private:

// Internal functions that help to implement the Standard Operations

     inline void _Exchange(FibHeapNode*&, FibHeapNode*&);
     void _Consolidate();
     void _Link(FibHeapNode *, FibHeapNode *);
     void _AddToRootList(FibHeapNode *);
     void _Cut(FibHeapNode *, FibHeapNode *);
     void _CascadingCut(FibHeapNode *);
};


class HeapNode : public FibHeapNode
{
public:
      double   N;
      long int IndexV;

  //public:

      HeapNode() : FibHeapNode() { N = 0; };   
      HeapNode(FibHeapNode fhn_) : FibHeapNode (fhn_) { N = 0; };   
      HeapNode(double N_, long int in_) : FibHeapNode() { N = N_; IndexV=in_; };   

      virtual void operator =(FibHeapNode& RHS);
      virtual int  operator ==(FibHeapNode& RHS);
      virtual int  operator <(FibHeapNode& RHS);

      virtual void operator =(double NewKeyVal );
  //      virtual void Print();

      double GetKeyValue() { return N; };       /* !!!! */
      void SetKeyValue(double n) { N = n; };

      long int GetIndexValue() { return IndexV; };
      void SetIndexValue( long int v) { IndexV = v; };

};


class Heap : public FibHeap
{
public:
     Heap() {};
     ~Heap() { while (GetNumNodes() > 0) ExtractMin(); }
     
     void Insert(double x, long int i) { FibHeap::Insert(new HeapNode(x,i));  }
     void Minimum(double& x, long int& i) { HeapNode* hn = static_cast<HeapNode*>(FibHeap::ExtractMin()); x=hn->N; i=hn->IndexV; }
     void ExtractMin(double& x, long int& i) { HeapNode* hn = static_cast<HeapNode*>(FibHeap::ExtractMin()); x=hn->N; i=hn->IndexV; delete hn; }
     void ExtractMin_safe (double& x, long int& i) { if (FibHeap::GetNumNodes()>0) {HeapNode* hn = static_cast<HeapNode*>(FibHeap::ExtractMin()); x=hn->N; i=hn->IndexV; delete hn;} }

 private:
     void ExtractMin() { delete (FibHeap::ExtractMin()); }     
};

#endif
