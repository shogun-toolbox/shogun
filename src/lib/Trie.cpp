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

CTrie::CTrie(const CTrie & to_copy)
	: degree(to_copy.degree), position_weights(NULL)
{
	if (to_copy.position_weights!=NULL)
	{
		position_weights = new DREAL[to_copy.length] ;
		for (INT i=0; i<to_copy.length; i++)
			position_weights[i]=to_copy.position_weights[i] ;
	}
	else
		position_weights=NULL ;
	
	TreeMemPtrMax=to_copy.TreeMemPtrMax ;
	TreeMemPtr=to_copy.TreeMemPtr ;
	TreeMem = (struct Trie*)malloc(TreeMemPtrMax*sizeof(struct Trie)) ;
	memcpy(TreeMem, to_copy.TreeMem, TreeMemPtrMax*sizeof(struct Trie)) ;
	
	length = to_copy.length ;
	trees=new INT[length] ;		
	for (INT i=0; i<length; i++)
		trees[i]=to_copy.trees[i] ;
	tree_initialized=to_copy.tree_initialized ;
}

const CTrie &CTrie::operator=(const CTrie & to_copy)
{
	degree=to_copy.degree ;
	delete[] position_weights ;
	position_weights=NULL ;
	if (to_copy.position_weights!=NULL)
	{
		position_weights = new DREAL[to_copy.length] ;
		for (INT i=0; i<to_copy.length; i++)
			position_weights[i]=to_copy.position_weights[i] ;
	}
	else
		position_weights=NULL ;
		
	TreeMemPtrMax=to_copy.TreeMemPtrMax ;
	TreeMemPtr=to_copy.TreeMemPtr ;
	free(TreeMem) ;
	TreeMem = (struct Trie*)malloc(TreeMemPtrMax*sizeof(struct Trie)) ;
	memcpy(TreeMem, to_copy.TreeMem, TreeMemPtrMax*sizeof(struct Trie)) ;
	
	length = to_copy.length ;
	if (trees)
		delete[] trees ;
	trees=new INT[length] ;		
	for (INT i=0; i<length; i++)
		trees[i]=to_copy.trees[i] ;
	tree_initialized=to_copy.tree_initialized ;

	return *this ;
}

INT CTrie::find_deepest_node(INT start_node, INT& deepest_node) const 
{
	INT ret=0 ;
	fprintf(stderr, "start_node=%i\n", start_node) ;
	
	if (start_node==NO_CHILD) 
	{
		for (INT i=0; i<length; i++)
		{
			INT my_deepest_node ;
			INT depth=find_deepest_node(i, my_deepest_node) ;
			fprintf(stderr, "start_node %i depth=%i\n", i, depth) ;
			if (depth>ret)
			{
				deepest_node=my_deepest_node ;
				ret=depth ;
			}
		}
		return ret ;
	}
	
	if (TreeMem[start_node].has_seq)
	{
		for (INT q=0; q<16; q++)
			if (TreeMem[start_node].seq[q]!=TRIE_TERMINAL_CHARACTER)
				ret++ ;
		deepest_node=start_node ;
		return ret ;
	}
	if (TreeMem[start_node].has_floats)
	{
		deepest_node=start_node ;
		return 1 ;
	}
	
	for (INT q=0; q<4; q++)
	{
		INT my_deepest_node ;
		if (TreeMem[start_node].children[q]==NO_CHILD)
			continue ;
		INT depth=find_deepest_node(abs(TreeMem[start_node].children[q]), my_deepest_node) ;
		if (depth>ret)
		{
			deepest_node=my_deepest_node ;
			ret=depth ;
		}
	}
	return ret ;
}


bool CTrie::compare_traverse(INT node, const CTrie & other, INT other_node) 
{
	fprintf(stderr, "checking nodes %i and %i\n", node, other_node) ;
	if (fabs(TreeMem[node].weight-other.TreeMem[other_node].weight)>=1e-5)
	{
		CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].weight=%f!=other.TreeMem[%i].weight=%f\n", node, TreeMem[node].weight, other_node,other.TreeMem[other_node].weight) ;
		CIO::message(M_DEBUG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
		display_node(node) ;
		CIO::message(M_DEBUG, "============================================================\n") ;			
		other.display_node(other_node) ;
		CIO::message(M_DEBUG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;			
		return false ;
	}
	
#ifdef TRIE_CHECK_EVERYTHING
	if (TreeMem[node].has_seq!=other.TreeMem[other_node].has_seq)
	{
		CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].has_seq=%i!=other.TreeMem[%i].has_seq=%i\n", node, TreeMem[node].has_seq, other_node,other.TreeMem[other_node].has_seq) ;
		CIO::message(M_DEBUG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
		display_node(node) ;
		CIO::message(M_DEBUG, "============================================================\n") ;			
		other.display_node(other_node) ;
		CIO::message(M_DEBUG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;
		return false ;
	}
	if (TreeMem[node].has_floats!=other.TreeMem[other_node].has_floats)
	{
		CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].has_floats=%i!=other.TreeMem[%i].has_floats=%i\n", node, TreeMem[node].has_floats, other_node, other.TreeMem[other_node].has_floats) ;
		return false ;
	}
	if (other.TreeMem[other_node].has_floats)
	{
		for (INT q=0; q<4; q++)
			if (fabs(TreeMem[node].child_weights[q]-other.TreeMem[other_node].child_weights[q])>1e-5)
			{
				CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].child_weights[%i]=%e!=other.TreeMem[%i].child_weights[%i]=%e\n", node, q,TreeMem[node].child_weights[q], other_node,q,other.TreeMem[other_node].child_weights[q]) ;
				CIO::message(M_DEBUG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
				display_node(node) ;
				CIO::message(M_DEBUG, "============================================================\n") ;			
				other.display_node(other_node) ;
				CIO::message(M_DEBUG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;			
				return false ;
			}
	}
	if (other.TreeMem[other_node].has_seq)
	{
		for (INT q=0; q<16; q++)
			if ((TreeMem[node].seq[q]!=other.TreeMem[other_node].seq[q]) && ((TreeMem[node].seq[q]<4)||(other.TreeMem[other_node].seq[q]<4)))
			{
				CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].seq[%i]=%i!=other.TreeMem[%i].seq[%i]=%i\n", node,q,TreeMem[node].seq[q], other_node,q,other.TreeMem[other_node].seq[q]) ;
				CIO::message(M_DEBUG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
				display_node(node) ;
				CIO::message(M_DEBUG, "============================================================\n") ;			
				other.display_node(other_node) ;
				CIO::message(M_DEBUG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;			
				return false ;
			}
	}
	if (!other.TreeMem[other_node].has_seq && !other.TreeMem[other_node].has_floats)
	{
		for (INT q=0; q<4; q++)
		{
			if ((TreeMem[node].children[q]==NO_CHILD) && (other.TreeMem[other_node].children[q]==NO_CHILD))
				continue ;
			if ((TreeMem[node].children[q]==NO_CHILD)!=(other.TreeMem[other_node].children[q]==NO_CHILD))
			{
				CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].children[%i]=%i!=other.TreeMem[%i].children[%i]=%i\n", node,q,TreeMem[node].children[q], other_node,q,other.TreeMem[other_node].children[q]) ;
				CIO::message(M_DEBUG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
				display_node(node) ;
				CIO::message(M_DEBUG, "============================================================\n") ;			
				other.display_node(other_node) ;
				CIO::message(M_DEBUG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;
				return false ;
			}
			if (!compare_traverse(abs(TreeMem[node].children[q]), other, abs(other.TreeMem[other_node].children[q])))
				return false ;
		}
	}

#endif
	
	return true ;
}

bool CTrie::compare(const CTrie & other)
{
	/*if (TreeMemPtr!=other.TreeMemPtr)
	{
		CIO::message(M_DEBUG, "CTrie::compare: TreeMemPtr=%i!=other.TreeMemPtr=%i\n", TreeMemPtr, other.TreeMemPtr) ;
		return false ;
	}
	if (length!=other.length)
	{
		CIO::message(M_DEBUG, "CTrie::compare: unequal number of trees\n") ;
		return false ;
	}
	if (tree_initialized!=other.tree_initialized)
	{
		CIO::message(M_DEBUG, "CTrie::compare: unequal initialized status\n") ;
		return false ;
		}*/
	
	bool ret=true ;
	for (INT i=0; i<length; i++)
		if (!compare_traverse(trees[i], other, other.trees[i]))
			return false ;
		else
			fprintf(stderr, "two tries at %i identical\n", i) ;

	/*
	for (INT i=0; i<TreeMemPtr; i++)
	{
		if (TreeMem[i].weight!=other.TreeMem[i].weight)
		{
			CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].weight=%f!=other.TreeMem[%i].weight=%f\n", i,TreeMem[i].weight,i,other.TreeMem[i].weight) ;
			CIO::message(M_DEBUG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
			display_node(i) ;
			CIO::message(M_DEBUG, "============================================================\n") ;			
			other.display_node(i) ;
			CIO::message(M_DEBUG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;			
			ret=false ;
			break ;
		}
#ifdef TRIE_CHECK_EVERYTHING
		if (TreeMem[i].has_seq!=other.TreeMem[i].has_seq)
		{
			CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].has_seq=%i!=other.TreeMem[%i].has_seq=%i\n", i,TreeMem[i].has_seq,i,other.TreeMem[i].has_seq) ;
			ret=false ;
		}
		if (TreeMem[i].has_floats!=other.TreeMem[i].has_floats)
		{
			CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].has_floats=%i!=other.TreeMem[%i].has_floats=%i\n", i,TreeMem[i].has_floats,i,other.TreeMem[i].has_floats) ;
			ret=false ;
		}
		if (other.TreeMem[i].has_floats)
		{
			for (INT q=0; q<4; q++)
				if (fabs(TreeMem[i].child_weights[q]-other.TreeMem[i].child_weights[q])>1e-6)
				{
					CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].child_weights[%i]=%e!=other.TreeMem[%i].child_weights[%i]=%e\n", i,q,TreeMem[i].child_weights[q],i,q,other.TreeMem[i].child_weights[q]) ;
					CIO::message(M_DEBUG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
					display_node(i) ;
					CIO::message(M_DEBUG, "============================================================\n") ;			
					other.display_node(i) ;
					CIO::message(M_DEBUG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;			
					ret=false ;
					break ;
				}
		}
		if (other.TreeMem[i].has_seq)
		{
			for (INT q=0; q<16; q++)
				if (TreeMem[i].seq[q]!=other.TreeMem[i].seq[q])
				{
					CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].seq[%i]=%i!=other.TreeMem[%i].seq[%i]=%i\n", i,q,TreeMem[i].seq[q],i,q,other.TreeMem[i].seq[q]) ;
					CIO::message(M_DEBUG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
					display_node(i) ;
					CIO::message(M_DEBUG, "============================================================\n") ;			
					other.display_node(i) ;
					CIO::message(M_DEBUG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;			
					ret=false ;
					break ;
				}
		}
		if (!other.TreeMem[i].has_seq && !other.TreeMem[i].has_floats)
		{
			for (INT q=0; q<4; q++)
				if (TreeMem[i].children[q]!=other.TreeMem[i].children[q])
				{
					CIO::message(M_DEBUG, "CTrie::compare: TreeMem[%i].children[%i]=%i!=other.TreeMem[%i].children[%i]=%i\n", i,q,TreeMem[i].children[q],i,q,other.TreeMem[i].children[q]) ;
					CIO::message(M_DEBUG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
					display_node(i) ;
					CIO::message(M_DEBUG, "============================================================\n") ;			
					other.display_node(i) ;
					CIO::message(M_DEBUG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;			
					ret=false ;
					break ;
				}
		}


#endif
	}
	if (ret)
		CIO::message(M_DEBUG, "CTrie::compare: no differences found\n") ;
	else
		CIO::message(M_DEBUG, "CTrie::compare: the tries differ\n") ;
	*/
	return ret ;
}

bool CTrie::find_node(INT node, INT * trace, INT& trace_len) const 
{
	ASSERT(trace_len-1>=0) ;
	//fprintf(stderr, "node=%i, trace_len=%i, trace=%i\n", node, trace_len, trace[trace_len-1]) ;
	ASSERT((trace[trace_len-1]>=0) && (trace[trace_len-1]<TreeMemPtrMax))
	if (TreeMem[trace[trace_len-1]].has_seq)
		return false ;
	if (TreeMem[trace[trace_len-1]].has_floats)
		return false ;

	for (INT q=0; q<4; q++)
	{
		if (TreeMem[trace[trace_len-1]].children[q]==NO_CHILD)
			continue ;
		INT tl=trace_len+1 ;
		if (TreeMem[trace[trace_len-1]].children[q]>=0)
			trace[trace_len]=TreeMem[trace[trace_len-1]].children[q] ;
		else
			trace[trace_len]=-TreeMem[trace[trace_len-1]].children[q] ;

		if (trace[trace_len]==node)
		{
			trace_len=tl ;
			return true ;
		}
		if (find_node(node, trace, tl))
		{
			trace_len=tl ;
			return true ;
		}
	}
	trace_len=0 ;
	return false ;
}

void CTrie::display_node(INT node) const
{
	INT * trace=new INT[2*degree] ;
	INT trace_len=-1 ;
	bool found = false ;
	INT tree=-1 ;
	for (tree=0; tree<length; tree++)
	{
		trace[0]=trees[tree] ;
		trace_len=1 ;
		found=find_node(node, trace, trace_len) ;
		if (found)
			break ;
	}
	ASSERT(found) ;
	CIO::message(M_MESSAGEONLY, "position %i  trace: ", tree) ;
	
	for (INT i=0; i<trace_len-1; i++)
	{
		INT branch=-1 ;
		for (INT q=0; q<4; q++)
			if (abs(TreeMem[trace[i]].children[q])==trace[i+1])
			{
				branch=q;
				break ;
			}
		ASSERT(branch!=-1) ;
		char acgt[5]="ACGT" ;
		CIO::message(M_MESSAGEONLY, "%c", acgt[branch]) ;
	}
	CIO::message(M_MESSAGEONLY, "\nnode=%i\nweight=%f\nhas_seq=%i\nhas_floats=%i\n", node, TreeMem[node].weight, TreeMem[node].has_seq, TreeMem[node].has_floats) ;
	if (TreeMem[node].has_floats)
	{
		for (INT q=0; q<4; q++)
			CIO::message(M_MESSAGEONLY, "child_weighs[%i] = %f\n", q, TreeMem[node].child_weights[q]) ;
	}
	if (TreeMem[node].has_seq)
	{
		for (INT q=0; q<16; q++)
			CIO::message(M_MESSAGEONLY, "seq[%i] = %i\n", q, TreeMem[node].seq[q]) ;
	}
	if (!TreeMem[node].has_seq && !TreeMem[node].has_floats)
	{
		for (INT q=0; q<4; q++)
		{
			if (TreeMem[node].children[q]!=NO_CHILD)
			{
				CIO::message(M_MESSAGEONLY, "children[%i] = %i -> \n", q, TreeMem[node].children[q]) ;
				display_node(abs(TreeMem[node].children[q])) ;
			}
			else
				CIO::message(M_MESSAGEONLY, "children[%i] = NO_CHILD -| \n", q, TreeMem[node].children[q]) ;
		}
		
	}
	
	delete[] trace ;
}


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
