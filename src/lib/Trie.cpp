#include "lib/common.h"
#include "lib/io.h"
#include "lib/Trie.h"
#include "lib/Mathematics.h"

CTrie::CTrie(INT d): degree(d), position_weights(NULL)
{
	TreeMemPtrMax=1024*1024/sizeof(struct Trie) ;
	TreeMemPtr=0 ;
	TreeMem = (struct Trie*)malloc(TreeMemPtrMax*sizeof(struct Trie)) ;
	
	length = 0;
	trees=NULL;
	tree_initialized=false ;
} ;


CTrie::~CTrie()
{
	destroy() ;
	
	free(TreeMem) ;
}

void CTrie::destroy()
{
	if (trees!=NULL)
	{
		delete_trees() ;
		for (INT i=0; i<length; i++)
			trees[i] = NO_CHILD;
		TreeMemPtr=0 ;
		delete[] trees;
		trees=NULL;
	}
}

void CTrie::create(INT len)
{
	trees=new INT[len] ;		
	TreeMemPtr=0 ;
	for (INT i=0; i<len; i++)
		trees[i]=get_node() ;
	length = len ;
}


void CTrie::delete_trees()
{
	if (trees==NULL)
		return;

	TreeMemPtr=0 ;
	for (INT i=0; i<length; i++)
		trees[i]=get_node() ;
} 

DREAL CTrie::compute_abs_weights_tree(INT tree, INT depth) 
{
  DREAL ret=0 ;
  
  if (tree==NO_CHILD)
    return 0 ;
  TRIE_ASSERT(tree>=0) ;
  
  if (depth==degree-2)
    {
      ret+=(TreeMem[tree].weight) ;
      
      for (INT k=0; k<4; k++)
	ret+=(TreeMem[tree].child_weights[k]) ;
      
      return ret ;
    }
  
  ret+=(TreeMem[tree].weight) ;
  
  for (INT i=0; i<4; i++)
    if (TreeMem[tree].children[i]!=NO_CHILD)
      ret += compute_abs_weights_tree(TreeMem[tree].children[i], depth+1)  ;
  
  return ret ;
}


DREAL *CTrie::compute_abs_weights(int &len) 
{
  DREAL * sum=new DREAL[length*4] ;
  for (INT i=0; i<length*4; i++)
    sum[i]=0 ;
  len=length ;
  
  for (INT i=0; i<length; i++)
    {
      TRIE_ASSERT(trees[i]!=NO_CHILD) ;
      for (INT k=0; k<4; k++)
	{
	  sum[i*4+k]=compute_abs_weights_tree(TreeMem[trees[i]].children[k], 0) ;
	}
    }
  
  return sum ;
}

void CTrie::compute_scoring_helper(INT tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result)
{
  if (tree==NO_CHILD)
    tree=trees[i] ;
  
  TRIE_ASSERT(tree!=NO_CHILD) ;
  
  if (i+j<num_feat)
  {
      if (j<degree-1)
	  {
		  for (INT k=0; k<num_sym; k++)
		  {
			  if (TreeMem[tree].children[k]!=NO_CHILD)
			  {
				  INT child=TreeMem[tree].children[k];
				  //continue recursion if not yet at max_degree, else add to result
				  if (d<max_degree-1)
					  compute_scoring_helper(child, i, j+1, weight+TreeMem[child].weight, d+1, max_degree, num_feat, num_sym, sym_offset, num_sym*offs+k, result);
				  else
					  result[sym_offset*(i+j-max_degree+1)+num_sym*offs+k] += weight;
				  
				  //do recursion starting from this position
				  if (d==0)
					  compute_scoring_helper(child, i, j+1, 0.0, 0, max_degree, num_feat, num_sym, sym_offset, offs, result);
			  }
		  }
	  }
      else if (j==degree-1)
	  {
		  for (INT k=0; k<num_sym; k++)
		  {
			  //continue recursion if not yet at max_degree, else add to result
			  if (d<max_degree-1 && i<num_feat-1)
				  compute_scoring_helper(trees[i+1], i+1, 0, weight+TreeMem[tree].child_weights[k], d+1, max_degree, num_feat, num_sym, sym_offset, num_sym*offs+k, result);
			  else
				  result[sym_offset*(i+j-max_degree+1)+num_sym*offs+k] += weight+TreeMem[tree].child_weights[k];
		  }
	  }
  }
}

void CTrie::add_example_to_tree_mismatch_recursion(INT tree,  INT i, DREAL alpha,
						   INT *vec, INT len_rem, 
						   INT degree_rec, INT mismatch_rec, 
						   INT max_mismatch, DREAL * weights) 
{
	if (tree==NO_CHILD)
		tree=trees[i] ;
	TRIE_ASSERT(tree!=NO_CHILD) ;
	
	if ((len_rem<=0) || (mismatch_rec>max_mismatch) || (degree_rec>degree))
		return ;
	const INT other[4][3] = {	{1,2,3},{0,2,3},{0,1,3},{0,1,2} } ;
	
	INT subtree = NO_CHILD ;
	
	if (degree_rec==degree-1)
    {
		TreeMem[tree].child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec];
		if (mismatch_rec+1<=max_mismatch)
			for (INT o=0; o<3; o++)
				TreeMem[tree].child_weights[other[vec[0]][o]] += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
		return ;
    }
	else
    {
		if (TreeMem[tree].children[vec[0]]!=NO_CHILD)
		{
			subtree=TreeMem[tree].children[vec[0]] ;
			TreeMem[subtree].weight += alpha*weights[degree_rec+degree*mismatch_rec];
		}
		else 
		{
			TreeMem[tree].children[vec[0]]=TreeMemPtr++ ;
			INT tmp=TreeMem[tree].children[vec[0]] ;
			check_treemem() ;
			subtree=tmp ;
			if (degree_rec==degree-2)
			{
				for (INT k=0; k<4; k++)
					TreeMem[tree].child_weights[k]=0;
			}
			else
			{
				for (INT k=0; k<4; k++)
					TreeMem[tree].children[k]=NO_CHILD;
			}
			TreeMem[subtree].weight = alpha*weights[degree_rec+degree*mismatch_rec] ;
		}
		add_example_to_tree_mismatch_recursion(subtree,  i, alpha,
											   &vec[1], len_rem-1, 
											   degree_rec+1, mismatch_rec, max_mismatch, weights) ;
		
		if (mismatch_rec+1<=max_mismatch)
		{
			for (INT o=0; o<3; o++)
			{
				INT ot = other[vec[0]][o] ;
				if (TreeMem[tree].children[ot]!=NO_CHILD)
				{
					subtree=TreeMem[tree].children[ot] ;
					TreeMem[subtree].weight += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
				}
				else 
				{
					TreeMem[tree].children[ot]=TreeMemPtr++ ;
					INT tmp=TreeMem[tree].children[ot] ;
					check_treemem() ;
					subtree=tmp ;
					if (degree_rec==degree-2)
					{
						for (INT k=0; k<4; k++)
							TreeMem[tree].child_weights[k]=0;
					}
					else
					{
						for (INT k=0; k<4; k++)
							TreeMem[tree].children[k]=NO_CHILD;
					}
					TreeMem[subtree].weight = alpha*weights[degree_rec+degree*(mismatch_rec+1)] ;
				}
				
				add_example_to_tree_mismatch_recursion(subtree,  i, alpha,
													   &vec[1], len_rem-1, 
													   degree_rec+1, mismatch_rec+1, max_mismatch, weights) ;
			}
		}
    }
}
