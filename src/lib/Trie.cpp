#include "lib/common.h"
#include "lib/io.h"
#include "lib/Trie.h"
#include "lib/Mathematics.h"

CTrie::CTrie(INT d, INT p_use_compact_terminal_nodes)
	: degree(d), position_weights(NULL), use_compact_terminal_nodes(p_use_compact_terminal_nodes)
{
	TreeMemPtrMax=1024*1024/sizeof(struct Trie) ;
	TreeMemPtr=0 ;
	TreeMem = (struct Trie*)malloc(TreeMemPtrMax*sizeof(struct Trie)) ;
	
	length = 0;
	trees=NULL;
	tree_initialized=false ;
} ;

CTrie::CTrie(const CTrie & to_copy)
	: degree(to_copy.degree), position_weights(NULL), use_compact_terminal_nodes(to_copy.use_compact_terminal_nodes)
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
	use_compact_terminal_nodes=to_copy.use_compact_terminal_nodes ;
	
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
#ifdef TRIE_CHECK_EVERYTHING
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
#else
	CIO::message(M_ERROR, "not implemented\n") ;
	return 0 ;
#endif
}

INT CTrie::compact_nodes(INT start_node, INT depth, DREAL * weights) 
{
	CIO::message(M_ERROR, "code buggy\n") ;
	
	INT ret=0 ;
	//fprintf(stderr, "start_node=%i\n", start_node) ;
	
	if (start_node==NO_CHILD) 
	{
		for (INT i=0; i<length; i++)
			compact_nodes(i,1, weights) ;
		return 0 ;
	}
	if (start_node<0)
		return -1 ;

	if (depth==degree-1)
	{
		TRIE_ASSERT_EVERYTHING(TreeMem[start_node].has_floats) ;
		INT num_used=0 ;
		for (INT q=0; q<4; q++)
			if (TreeMem[start_node].child_weights[q]!=0.0)
				num_used++ ;
		if (num_used>1)
			return -1 ;
		return 1 ;
	}
	TRIE_ASSERT_EVERYTHING(!TreeMem[start_node].has_floats) ;
	
	INT num_used = 0 ;
	INT q_used=-1 ;
	
	for (INT q=0; q<4; q++)
	{
		if (TreeMem[start_node].children[q]==NO_CHILD)
			continue ;
		num_used++ ;
		q_used=q ;
	}
	if (num_used>1)
	{
		if (depth>=degree-2)
			return -1 ;
		for (INT q=0; q<4; q++)
		{
			if (TreeMem[start_node].children[q]==NO_CHILD)
				continue ;
			INT num=compact_nodes(abs(TreeMem[start_node].children[q]), depth+1, weights) ;
			if (num<=2)
				continue ;
			INT node=get_node() ;
			//fprintf(stderr, "creating node:\n ") ;
			INT last_node=TreeMem[start_node].children[q] ;
#ifdef WEIGHTS_IN_TRIE 
			ASSERT(weights[depth]!=0.0) ;
			TreeMem[node].weight=TreeMem[last_node].weight/weights[depth] ;
#else
			TreeMem[node].weight=TreeMem[last_node].weight ;
#endif
#ifdef TRIE_CHECK_EVERYTHING
			TreeMem[node].has_seq=true ;
#endif
			memset(TreeMem[node].seq, TRIE_TERMINAL_CHARACTER, 16) ;
			for (INT n=0; n<num; n++)
			{
				ASSERT(depth+n+1<=degree-1) ;
				ASSERT(last_node!=NO_CHILD) ;
				if (depth+n+1==degree-1)
				{
					TRIE_ASSERT_EVERYTHING(TreeMem[last_node].has_floats) ;
					INT  k ;
					for (k=0; k<4; k++)
						if (TreeMem[last_node].child_weights[k]!=0.0)
							break ;
					if (k==4)
						break ;
					TreeMem[node].seq[n]=k ;
					//fprintf(stderr, "seq[%i]=%i\n", n, k) ;
					break ;
				}
				else
				{
					TRIE_ASSERT_EVERYTHING(!TreeMem[last_node].has_floats) ;
					INT k ;
					for (k=0; k<4; k++)
						if (TreeMem[last_node].children[k]!=NO_CHILD)
							break ;
					if (k==4)
						break ;
					TreeMem[node].seq[n]=k ;
					//fprintf(stderr, "seq[%i]=%i\n", n, k) ;
					last_node=TreeMem[last_node].children[k] ;
				}
			}
			TreeMem[start_node].children[q]=-node ;
		}
		return -1 ;
	}
	if (num_used==0)
		return 0 ;
	
	ret=compact_nodes(abs(TreeMem[start_node].children[q_used]), depth+1, weights) ;
	if (ret<0)
		return ret ;
	return ret+1 ;
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
#else
	CIO::message(M_ERROR, "not implemented\n") ;
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

	return ret ;
}

bool CTrie::find_node(INT node, INT * trace, INT& trace_len) const 
{
#ifdef TRIE_CHECK_EVERYTHING
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
#else
	CIO::message(M_ERROR, "not implemented\n") ;
	return false ;
#endif
}

void CTrie::display_node(INT node) const
{
#ifdef TRIE_CHECK_EVERYTHING
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
#else
	CIO::message(M_ERROR, "not implemented\n") ;
#endif
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

void CTrie::create(INT len, INT p_use_compact_terminal_nodes)
{
	if (trees)
		delete[] trees ;
	
	trees=new INT[len] ;		
	TreeMemPtr=0 ;
	for (INT i=0; i<len; i++)
		trees[i]=get_node() ;
	length = len ;

	use_compact_terminal_nodes=p_use_compact_terminal_nodes ;
}


void CTrie::delete_trees(INT p_use_compact_terminal_nodes)
{
	if (trees==NULL)
		return;

	TreeMemPtr=0 ;
	for (INT i=0; i<length; i++)
		trees[i]=get_node() ;

	use_compact_terminal_nodes=p_use_compact_terminal_nodes ;
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
		TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_floats) ;
#ifdef WEIGHTS_IN_TRIE 
		TreeMem[tree].child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec];
#else
		if (weights[degree_rec]!=0.0)
			TreeMem[tree].child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec]/weights[degree_rec];
#endif
		if (mismatch_rec+1<=max_mismatch)
			for (INT o=0; o<3; o++)
			{
#ifdef WEIGHTS_IN_TRIE 
				TreeMem[tree].child_weights[other[vec[0]][o]] += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
#else
				if (weights[degree_rec]!=0.0)
					TreeMem[tree].child_weights[other[vec[0]][o]] += alpha*weights[degree_rec+degree*(mismatch_rec+1)]/weights[degree_rec];
#endif
			}
		return ;
    }
	else
    {
		TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;
		if (TreeMem[tree].children[vec[0]]!=NO_CHILD)
		{
			subtree=TreeMem[tree].children[vec[0]] ;
#ifdef WEIGHTS_IN_TRIE 
			TreeMem[subtree].weight += alpha*weights[degree_rec+degree*mismatch_rec];
#else
			if (weights[degree_rec]!=0.0)
				TreeMem[subtree].weight += alpha*weights[degree_rec+degree*mismatch_rec]/weights[degree_rec];
#endif
		}
		else 
		{
			INT tmp = get_node() ;
			ASSERT(tmp>=0) ;
			TreeMem[tree].children[vec[0]]=tmp ;
			subtree=tmp ;
			if (degree_rec==degree-2)
			{
#ifdef TRIE_CHECK_EVERYTHING
				TreeMem[subtree].has_floats=true ;
#endif
				for (INT k=0; k<4; k++)
					TreeMem[subtree].child_weights[k]=0;
			}
			else
			{
				for (INT k=0; k<4; k++)
					TreeMem[subtree].children[k]=NO_CHILD;
			}
#ifdef WEIGHTS_IN_TRIE 
			TreeMem[subtree].weight = alpha*weights[degree_rec+degree*mismatch_rec] ;
#else
			if (weights[degree_rec]!=0.0)
				TreeMem[subtree].weight = alpha*weights[degree_rec+degree*mismatch_rec]/weights[degree_rec] ;
			else
				TreeMem[subtree].weight = 0.0 ;
#endif
		}
		add_example_to_tree_mismatch_recursion(subtree,  i, alpha,
											   &vec[1], len_rem-1, 
											   degree_rec+1, mismatch_rec, max_mismatch, weights) ;
		
		if (mismatch_rec+1<=max_mismatch)
		{
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;
			for (INT o=0; o<3; o++)
			{
				INT ot = other[vec[0]][o] ;
				if (TreeMem[tree].children[ot]!=NO_CHILD)
				{
					subtree=TreeMem[tree].children[ot] ;
#ifdef WEIGHTS_IN_TRIE 
					TreeMem[subtree].weight += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
#else
					if (weights[degree_rec]!=0.0)
						TreeMem[subtree].weight += alpha*weights[degree_rec+degree*(mismatch_rec+1)]/weights[degree_rec];
#endif
				}
				else 
				{
					INT tmp = get_node() ;
					ASSERT(tmp>=0) ;
					TreeMem[tree].children[ot]=tmp ;
					subtree=tmp ;
					if (degree_rec==degree-2)
					{
#ifdef TRIE_CHECK_EVERYTHING
						TreeMem[subtree].has_floats=true ;
#endif
						for (INT k=0; k<4; k++)
							TreeMem[subtree].child_weights[k]=0;
					}
					else
					{
						for (INT k=0; k<4; k++)
							TreeMem[subtree].children[k]=NO_CHILD;
					}
#ifdef WEIGHTS_IN_TRIE 
					TreeMem[subtree].weight = alpha*weights[degree_rec+degree*(mismatch_rec+1)] ;
#else
					if (weights[degree_rec]!=0.0)
						TreeMem[subtree].weight = alpha*weights[degree_rec+degree*(mismatch_rec+1)]/weights[degree_rec] ;
					else
						TreeMem[subtree].weight = 0.0 ;
#endif
				}
				
				add_example_to_tree_mismatch_recursion(subtree,  i, alpha,
													   &vec[1], len_rem-1, 
													   degree_rec+1, mismatch_rec+1, max_mismatch, weights) ;
			}
		}
    }
}

void CTrie::compute_scoring_helper(INT tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result)
{
	if (i+j<num_feat)
    {
		DREAL decay=1.0; //no decay by default
		//if (j>d)
		//	decay=pow(0.5,j); //marginalize out lower order matches
		
		if (j<degree-1)
		{
			for (INT k=0; k<num_sym; k++)
			{
				if (TreeMem[tree].children[k]!=NO_CHILD)
				{
					INT child=TreeMem[tree].children[k];
					//continue recursion if not yet at max_degree, else add to result
					if (d<max_degree-1)
						compute_scoring_helper(child, i, j+1, weight+decay*TreeMem[child].weight, d+1, max_degree, num_feat, num_sym, sym_offset, num_sym*offs+k, result);
					else
						result[sym_offset*(i+j-max_degree+1)+num_sym*offs+k] += weight+decay*TreeMem[child].weight;
					
					////do recursion starting from this position
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
					compute_scoring_helper(trees[i+1], i+1, 0, weight+decay*TreeMem[tree].child_weights[k], d+1, max_degree, num_feat, num_sym, sym_offset, num_sym*offs+k, result);
				else
					result[sym_offset*(i+j-max_degree+1)+num_sym*offs+k] += weight+decay*TreeMem[tree].child_weights[k];
			}
		}
    }
}

void CTrie::traverse( INT tree, const INT p, struct TreeParseInfo info, const INT depth, INT* const x, const INT k )
{
    const INT num_sym = info.num_sym;
    const INT y0 = info.y0;
    const INT y1 = (k==0) ? 0 : y0 - ( (depth<k) ? 0 : info.nofsKmers[k-1] * x[depth-k] );
    //const INT temp = info.substrs[depth]*num_sym - ( (depth<=k) ? 0 : info.nofsKmers[k] * x[depth-k-1] );
    //if( !( info.y0 == temp ) ) {
    //  printf( "\n temp=%d y0=%d k=%d depth=%d \n", temp, info.y0, k, depth );
    //}
    //ASSERT( info.y0 == temp );
    INT sym;
    ASSERT( depth < degree );
    //ASSERT( 0 <= info.substrs[depth] && info.substrs[depth] < info.nofsKmers[k] );
    if (depth<degree-1)
    {
		for( sym=0; sym<num_sym; ++sym ) {
			const INT childNum = TreeMem[tree].children[ sym ];
			if( childNum != NO_CHILD ) {
				INT child = childNum ;
				x[depth] = sym;
				info.substrs[depth+1] = y0 + sym;
				info.y0 = (k==0) ? 0 : (y1+sym)*num_sym;
				//ASSERT( info.y0 == ( info.substrs[depth+1]*num_sym - ( (depth<k) ? 0 : info.nofsKmers[k] * x[depth-k] ) ) );
				count( TreeMem[child].weight, depth, info, p, x, k );
				traverse( child, p, info, depth+1, x, k );
				x[depth] = -1;
			}
		}
    }
    else if( depth == degree-1 )
    {
        for( sym=0; sym<num_sym; ++sym ) {
			const DREAL w = TreeMem[tree].child_weights[ sym ];
			if( w != 0.0 ) {
				x[depth] = sym;
				info.substrs[depth+1] = y0 + sym;
				info.y0 = (k==0) ? 0 : (y1+sym)*num_sym;
				//ASSERT( info.y0 == ( info.substrs[depth+1]*num_sym - ( (depth<k) ? 0 : info.nofsKmers[k] * x[depth-k] ) ) );
				count( w, depth, info, p, x, k );
				x[depth] = -1;
			}
		}
    }
    //info.substrs[depth+1] = -1;
    //info.y0 = temp;
}

void CTrie::count( const DREAL w, const INT depth, const struct TreeParseInfo info, const INT p, INT* x, const INT k )
{
    ASSERT( fabs(w) < 1e10 );
    ASSERT( x[depth] >= 0 );
    ASSERT( x[depth+1] < 0 );
    if ( depth < k ) {
		return;
    }
    //ASSERT( info.margFactors[ depth-k ] == pow( 0.25, depth-k ) );
    const INT nofKmers = info.nofsKmers[k];
    const DREAL margWeight =  w * info.margFactors[ depth-k ];
    const INT m_a = depth - k + 1;
    const INT m_b = info.num_feat - p;
    const INT m = ( m_a < m_b ) ? m_a : m_b;
    // all proper k-substrings
    const INT offset0 = nofKmers * p;
    register INT i;
    register INT offset;
    offset = offset0;
    for( i = 0; i < m; ++i ) {
        const INT y = info.substrs[i+k+1];
		info.C_k[ y + offset ] += margWeight;
		offset += nofKmers;
    }
    if( depth > k ) {
		// k-prefix
		const INT offsR = info.substrs[k+1] + offset0;
		info.R_k[offsR] += margWeight;
		// k-suffix
		if( p+depth-k < info.num_feat ) {
			const INT offsL = info.substrs[depth+1] + nofKmers * (p+depth-k);
			info.L_k[offsL] += margWeight; 
		}
    }
    //    # N.x = substring represented by N
    //    # N.d = length of N.x
    //    # N.s = starting position of N.x
    //    # N.w = weight for feature represented by N
    //    if( N.d >= k )
    //      margContrib = w / 4^(N.d-k)
    //      for i = 1 to (N.d-k+1)
    //        y = N.x[i:(i+k-1)]  # overlapped k-mer
    //        C_k[ N.s+i-1, y ] += margContrib
    //      end;
    //      if( N.d > k )
    //        L_k[ N.s+N.d-k, N.x[N.d-k+(1:k)] ] += margContrib  # j-suffix of N.x
    //        R_k[ N.s,       N.x[1:k]         ] += margContrib  # j-prefix of N.x
    //      end;
    //    end;
}

