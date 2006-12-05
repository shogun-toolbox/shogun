#include "kernel/SpectrumKernel.h"
#include <queue>
#include <vector>
#include <algorithm>
#include <numeric>

CSpectrumKernel::CSpectrumKernel(CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r, INT cachesize) : CStringKernel<CHAR>(cachesize)
{
	esa = NULL;
	weigher = NULL;
	val = NULL;
	lvs = NULL;

	//weigher = new ConstantWeight();
	//weigher = new ExpDecayWeight();
	//weigher = new BinKSpecWeight();
	//weigher = new BoundedRangeWeight();
	weigher = new KSpectrumWeight();

	init(l,r,true);
}

bool CSpectrumKernel::init(CFeatures* lhs, CFeatures* rhs, bool force)
{
	INT verb=0;
	LONG len=0;
	for (int i=0; i<lhs->get_num_vectors(); i++)
		len += ((CStringFeatures<CHAR>*) lhs)->get_vector_length(i)+1;

	ASSERT(len>0);

	SYMBOL* text=new SYMBOL[len];

	LONG offs=0;
	for (int i=0; i<((CStringFeatures<CHAR>*) lhs)->get_num_vectors(); i++)
	{
		INT len=0;
		CHAR* str=((CStringFeatures<CHAR>*) lhs)->get_feature_vector(i,len);

		ASSERT(len>0);
		memcpy(text+offs, str, len);
		text[len]='\n';
		offs+=len+1;
	}

	//' Build ESA. 
	esa = new ESA(len, text, verb);

	//' Allocate memory space for #val#
	val = new Real[esa->size+1];

	//' Allocate memory space for #lvs#
	//lvs = new Real[esa->size+1];
	lvs = NULL;
	
	return true;
}

CSpectrumKernel::CSpectrumKernel(INT cachesize) : CStringKernel<CHAR>(cachesize)
{
	esa = NULL;
	weigher = NULL;
	val = NULL;
	lvs = NULL;

	//weigher = new ConstantWeight();
	//weigher = new ExpDecayWeight();
	//weigher = new BinKSpecWeight();
	//weigher = new BoundedRangeWeight();
	weigher = new KSpectrumWeight();
}

CSpectrumKernel::~CSpectrumKernel()
{
	cleanup();
}

void CSpectrumKernel::cleanup()
{
	//' Delete objects and release allocated memory space.
	if(esa) { delete esa; esa = NULL; }
	if(val) { delete [] val; val = NULL; }
	if(lvs) { delete [] lvs; lvs = NULL; }
	if(weigher) { delete weigher; weigher = NULL; }
}


/**
 *  An Iterative auxiliary function used in PrecomputeVal().
 *
 *  Note: Every lcp-interval can be represented by its first l-index.
 *          Hence, 'val' is stored in val[] at the index := first l-index.
 *
 *  Pre: val[] is initialised to 0.
 *
 *  \param left    - (IN) Left bound of current interval.
 *  \param right   - (IN) Right bound of current interval.
 */
ErrorCode
CSpectrumKernel::IterativeCompute(const UInt32 &left, const UInt32 &right)
{
	using namespace std;
	ErrorCode ec;

 	//' Variables
	queue<pair<UInt32,UInt32> > q;
	vector<pair<UInt32,UInt32> > childlist;
	pair<UInt32,UInt32> p;
	UInt32 lb = 0;
	UInt32 rb = 0;
	UInt32 floor_len = 0;
	UInt32 x_len = 0;
	Real cur_val = 0.0;
	Real edge_weight = 0.0;


	//' Step 1: At root, 0-[0..size-1]. Store all non-single child-intervals onto #q#.
	lb = left;   //' Should be equal to 0.
	rb = right;  //' Should be equal to size-1.
	ec = esa->GetChildIntervals(lb,rb,childlist); CHECKERROR(ec);

	for(UInt32 jj=0; jj<childlist.size(); jj++)
		q.push(childlist[jj]);


	//' Step 2: Do breadth-first traversal. For every interval, compute val and add
	//'           it to all its non-singleton child-intervals' val-entries in val[].
	//'         Start with child-interval [i..j] of 0-[0..size-1].
	//'         ASSERT(j != size-1)
	while(!q.empty()) {
		//' Step 2.1: Get an interval from queue, #q#.
		p = q.front(); q.pop();

		//' step 2.2: Get the lcp of floor interval.
		UInt32 a=0, b=0;

		a = esa->lcptab[p.first];
		//svnvish: BUGBUG
		// Glorious hack. We have to remove it later.
		// This gives the lcp of parent interval
		if(p.second < esa->size-1){ 
			b = esa->lcptab[p.second+1];
		}else{
			b = 0;
		}
		
		floor_len = (a>b) ? a : b;
				

		//' Step 2.3: Get the lcp of current interval.
		ec = esa->GetLcp(p.first,p.second,x_len); CHECKERROR(ec);
		

		//' Step 2.4: Compute val of current interval.
		ec = weigher->ComputeWeight(floor_len, x_len, edge_weight); CHECKERROR(ec);
		cur_val = edge_weight*(lvs[p.second+1]-lvs[p.first]);
			

		//' Step 2.5: Add #cur_val# to val[].
		UInt32 firstlIndex1 = 0;
		ec = esa->childtab.l_idx(p.first, p.second, firstlIndex1); CHECKERROR(ec);
		val[firstlIndex1] += cur_val;
	

		//' Step 2.6: Get all child-intervals of this interval.
		childlist.clear();
		ec = esa->GetChildIntervals(p.first,p.second,childlist); CHECKERROR(ec);
		

		//' Step 2.7: (a) Add #cur_val# to child-intervals' val-entries in val[].
		//'           (b) Push child-interval onto #q#.
		for(UInt32 kk=0; kk<childlist.size(); kk++) {
			//' (a)
			UInt32 firstlIndex2=0;
			pair<UInt32,UInt32> tmp_p = childlist[kk];
			
			if(esa->text[esa->suftab[tmp_p.first]] == SENTINEL)
				continue;

			ec = esa->childtab.l_idx(tmp_p.first, tmp_p.second, firstlIndex2);
			CHECKERROR(ec);
			// ASSERT( val[firstlIndex2] == 0 );
			val[firstlIndex2] = val[firstlIndex1]; // cur_val;
						
			//' (b)
			q.push(make_pair(tmp_p.first, tmp_p.second));
		}
	}
		
	return NOERROR;
}



/**
 *  Precomputation of val(t) of string kernel. 
 *  Observation :Every internal node of a suffix tree can be represented by at
 *                 least one index of the corresponding lcp array. So, the val
 *                 of a node is stored in val[] at the index equal to that of 
 *                 the fist representative lcp value in lcp[].
 */
ErrorCode
CSpectrumKernel::PrecomputeVal()
{
	ErrorCode ec;

	//' Memory space requirement check.
	ASSERT(val != NULL);


	//' Initialise all val entries to zero!
	memset(val,0,sizeof(Real)*esa->size+1);

	
	//' Start iterative precomputation of val[]
	ec = IterativeCompute(0,esa->size-1); CHECKERROR(ec);

	return NOERROR;
}


/**
 *  Compute k(text,x) by performing Chang and Lawler's matching statistics collection 
 *    algorithm on the enhanced suffix array.
 *
 *  \param x     - (IN) The input string which is to be evaluated together with
 *                        the text in esa.
 *  \param x_len - (IN) The length of #x#.
 *  \param value - (IN) The value of k(x,x').
 */
ErrorCode
CSpectrumKernel::Compute_K(SYMBOL *x, const UInt32 &x_len, Real &value)
{
	ErrorCode ec;

	//' Variables
	UInt32 floor_i = 0;
	UInt32 floor_j = 0;
	UInt32 i = 0;
	UInt32 j = 0;
	UInt32 lb = 0;
	UInt32 rb = 0;
	UInt32 matched_len = 0;
	UInt32 offset = 0;
	UInt32 floor_len = 0;
	UInt32 firstlIndex = 0;
	Real edge_weight = 0.0;


	//' Initialisation
	value = 0.0;
	lb = 0;
	rb = esa->size-1;

	
	//' for each suffix, xprime[k..xprime_len-1], find longest match in text
	for(UInt32 k=0; k < x_len; k++) {

		//' Step 1: Matching
		ec = esa->ExactSuffixMatch(lb, rb, offset, &x[k], x_len-k, i, j, matched_len,
															 floor_i, floor_j, floor_len); 
		CHECKERROR(ec);
		
			
		//' Step 2: Get suffix link for [floor_i..floor_j]
		ec = esa->GetSuflink(floor_i, floor_j, lb, rb); CHECKERROR(ec);
		ASSERT((floor_j-floor_i) <= (rb-lb));  //' Range check

				
		//' Step 3: Compute contribution of this matched substring
		ec = esa->childtab.l_idx(floor_i, floor_j, firstlIndex); CHECKERROR(ec);
		ASSERT(firstlIndex > floor_i && firstlIndex <= floor_j);
		ASSERT(floor_len <= matched_len);

		ec = weigher->ComputeWeight(floor_len, matched_len, edge_weight); CHECKERROR(ec);
		value += val[firstlIndex] + edge_weight*(lvs[j+1] - lvs[i]);


		//' Step 4: Prepare for next iteration.
		offset = (matched_len) ? matched_len-1 : 0;
		

	}//' for
	
	return NOERROR;
}



/**
 *  Assign leaf weights. 
 *
 *  Assumption: '\n' is only used in delimiting the concatenated strings.
 *                No string should contain '\n'.
 *
 *  \param leafWeight - (IN) The leaf weights (or \alpha in SVM??).
 *  \param len        - (IN) The array containing the length of each string in the Master String.
 *                             (Master String is the concatenation of '\n'-delimited strings.)
 *  \param m          - (IN) Size of the array #leafWeight# (equal to the number of datapoints).
 *
 */
ErrorCode
CSpectrumKernel::Set_Lvs(const Real *leafWeight, const UInt32 *len, const UInt32 &m)
{
	using namespace std;

	//' Clean up previous lvs, if any.
	if(lvs) {
		delete lvs;
		lvs = NULL;
	}
	
	//' Variables
	UInt32 pos = 0;

	
	//' Let n denotes the length of Master String, and
	//'     m denotes the number of strings in Master String.

 	//' n := sum{|string_i|} + m, where
	//'   m is the number of delimiter (i.e. '\n') added.
	

	//' Create a cumulative array of #len.
	UInt32 *clen = new (nothrow) UInt32[m];


	//' clen[] is a cumulative array of len[]
	partial_sum(len,len+m,clen);
	ASSERT(clen[m-1] == esa->size);


	//' Allocate memory space for lvs[]
	lvs = new (nothrow) Real[esa->size+1]; 
	ASSERT(lvs);


	//' Assign leaf weight to lvs element according to its position in text.
	for(UInt32 j=0; j < esa->size; j++) {
		pos = esa->suftab[j];
		
		UInt32 *p = upper_bound(clen, clen+m, pos);

		lvs[j+1] = leafWeight[p-clen];
	}

	
	//' Compute cumulative lvs[]. To be used in matching statistics computation later.
	lvs[0] = 0.0;
	partial_sum(lvs, lvs+esa->size+1, lvs);

	return NOERROR;
}


DREAL CSpectrumKernel::compute(INT idx_a, INT idx_b)
{
/**
 *  Construct string kernel when given only text and its length.
 *
 * \param text         - (IN) The text which SuffixArray and StringKernel correspond to.
 * \param text_length  - (IN) The length of #_text#.
 * \param verb         - (IN) Verbosity level.
 */

	return idx_a+idx_b;
}

/**
 *  Set lvs[i] = i, for i = 0 to esa->size
 *  Memory space for lvs[] will be allocated.
 */
ErrorCode
CSpectrumKernel::Set_Lvs()
{
	using namespace std;

	//' Clean up previous lvs, if any.
	if(lvs) {
		delete lvs;
		lvs = NULL;
	}

	//' Allocate memory space for lvs[]
	lvs = new (nothrow) Real[esa->size+1]; 
	
	//' Check if memory correctly allocated.
	ASSERT(lvs != NULL);

	//' Range := [0..esa->size]
	for(UInt32 i=0; i<= esa->size; i++)
		lvs[i] = i;

	return NOERROR;
}

bool CSpectrumKernel::load_init(FILE* src)
{
	return false;
}

bool CSpectrumKernel::save_init(FILE* dest)
{
	return false;
}
