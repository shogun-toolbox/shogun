/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, Wu Lin
 *          Jacob Walker, Evan Shelhamer, Giovanni De Toni, Viktor Gal,
 *          Roman Votyakov, Esben Sorig, Evgeniy Andreev, Fernando Iglesias,
 *          Saurabh Goyal, Shashwat Lal Das, Thoralf Klein, Hu Shell,
 *          Soumyajitde De, Evangelos Anagnostopoulos
 */

#include <shogun/base/progress.h>
#include <shogun/io/File.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

#include <shogun/base/Parallel.h>

#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/normalizer/IdentityKernelNormalizer.h>
#include <shogun/features/Features.h>

#include <shogun/classifier/svm/SVM.h>

#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <shogun/mathematics/Math.h>

using namespace shogun;

Kernel::Kernel() : SGObject()
{
	init();
	register_params();
}

Kernel::Kernel(int32_t size) : SGObject()
{
	init();

	if (size<10)
		size=10;

	cache_size=size;
	register_params();
}


Kernel::Kernel(std::shared_ptr<Features> p_lhs, std::shared_ptr<Features> p_rhs, int32_t size) : SGObject()
{
	init();

	if (size<10)
		size=10;

	cache_size=size;

	set_normalizer(std::make_shared<IdentityKernelNormalizer>());
	init(p_lhs, p_rhs);
	register_params();
}

Kernel::~Kernel()
{
	if (get_is_initialized())
		error("Kernel still initialized on destruction.");

	remove_lhs_and_rhs();

}

#ifdef USE_SVMLIGHT
void Kernel::resize_kernel_cache(KERNELCACHE_IDX size, bool regression_hack)
{
	if (size<10)
		size=10;

	kernel_cache_cleanup();
	cache_size=size;

	if (has_features() && get_num_vec_lhs())
		kernel_cache_init(cache_size, regression_hack);
}
#endif //USE_SVMLIGHT

bool Kernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	//make sure features were indeed supplied
	require(l, "Kernel::init({}, {}): Left hand side features required!", fmt::ptr(l.get()), fmt::ptr(r.get()));
	require(r, "Kernel::init({}, {}): Right hand side features required!", fmt::ptr(l.get()), fmt::ptr(r.get()));

	//make sure features are compatible
	if (l->support_compatible_class())
	{
		require(l->get_feature_class_compatibility(r->get_feature_class()),
			"Right hand side of features ({}) must be compatible with left hand side features ({})",
			l->get_name(), r->get_name());
	}
	else
	{
		require(l->get_feature_class()==r->get_feature_class(),
			"Right hand side of features ({}) must be compatible with left hand side features ({})",
			l->get_name(), r->get_name());
	}
	ASSERT(l->get_feature_type()==r->get_feature_type())

	//remove references to previous features
	remove_lhs_and_rhs();


	if (l==r)
		lhs_equals_rhs=true;


	lhs=l;
	rhs=r;

	ASSERT(!num_lhs || num_lhs==l->get_num_vectors())
	ASSERT(!num_rhs || num_rhs==l->get_num_vectors())

	num_lhs=l->get_num_vectors();
	num_rhs=r->get_num_vectors();

	SG_TRACE("leaving Kernel::init({}, {})", fmt::ptr(l.get()), fmt::ptr(r.get()));
	return true;
}

bool Kernel::set_normalizer(std::shared_ptr<KernelNormalizer> n)
{

	if (lhs && rhs)
		n->init(this);


	normalizer=n;

	return (normalizer!=NULL);
}

std::shared_ptr<KernelNormalizer> Kernel::get_normalizer() const
{

	return normalizer;
}

bool Kernel::init_normalizer()
{
	return normalizer->init(this);
}

void Kernel::cleanup()
{
	remove_lhs_and_rhs();
}

#ifdef USE_SVMLIGHT
/****************************** Cache handling *******************************/

void Kernel::kernel_cache_init(int32_t buffsize, bool regression_hack)
{
	int32_t totdoc=get_num_vec_lhs();
	if (totdoc<=0)
	{
		error("kernel has zero rows: num_lhs={} num_rhs={}",
				get_num_vec_lhs(), get_num_vec_rhs());
	}
	uint64_t buffer_size=0;
	int32_t i;

	//in regression the additional constraints are made by doubling the training data
	if (regression_hack)
		totdoc*=2;

	buffer_size=((uint64_t) buffsize)*1024*1024/sizeof(KERNELCACHE_ELEM);
	if (buffer_size>((uint64_t) totdoc)*totdoc)
		buffer_size=((uint64_t) totdoc)*totdoc;

	io::info("using a kernel cache of size {} MB ({} bytes) for {} Kernel", buffer_size*sizeof(KERNELCACHE_ELEM)/1024/1024, buffer_size*sizeof(KERNELCACHE_ELEM), get_name());

	//make sure it fits in the *signed* KERNELCACHE_IDX type
	ASSERT(buffer_size < (((uint64_t) 1) << (sizeof(KERNELCACHE_IDX)*8-1)))

	kernel_cache.index = SG_MALLOC(int32_t, totdoc);
	kernel_cache.occu = SG_MALLOC(int32_t, totdoc);
	kernel_cache.lru = SG_MALLOC(int32_t, totdoc);
	kernel_cache.invindex = SG_MALLOC(int32_t, totdoc);
	kernel_cache.active2totdoc = SG_MALLOC(int32_t, totdoc);
	kernel_cache.totdoc2active = SG_MALLOC(int32_t, totdoc);
	kernel_cache.buffer = SG_MALLOC(KERNELCACHE_ELEM, buffer_size);
	kernel_cache.buffsize=buffer_size;
	kernel_cache.max_elems=(int32_t) (kernel_cache.buffsize/totdoc);

	if(kernel_cache.max_elems>totdoc) {
		kernel_cache.max_elems=totdoc;
	}

	kernel_cache.elems=0;   // initialize cache
	for(i=0;i<totdoc;i++) {
		kernel_cache.index[i]=-1;
		kernel_cache.lru[i]=0;
	}
	for(i=0;i<totdoc;i++) {
		kernel_cache.occu[i]=0;
		kernel_cache.invindex[i]=-1;
	}

	kernel_cache.activenum=totdoc;;
	for(i=0;i<totdoc;i++) {
		kernel_cache.active2totdoc[i]=i;
		kernel_cache.totdoc2active[i]=i;
	}

	kernel_cache.time=0;
}

void Kernel::get_kernel_row(
	int32_t docnum, int32_t *active2dnum, float64_t *buffer, bool full_line)
{
	int32_t i,j;
	KERNELCACHE_IDX start;

	int32_t num_vectors = get_num_vec_lhs();
	if (docnum>=num_vectors)
		docnum=2*num_vectors-1-docnum;

	/* is cached? */
	if(kernel_cache.index[docnum] != -1)
	{
		kernel_cache.lru[kernel_cache.index[docnum]]=kernel_cache.time; /* lru */
		start=((KERNELCACHE_IDX) kernel_cache.activenum)*kernel_cache.index[docnum];

		if (full_line)
		{
			for(j=0;j<get_num_vec_lhs();j++)
			{
				if(kernel_cache.totdoc2active[j] >= 0)
					buffer[j]=kernel_cache.buffer[start+kernel_cache.totdoc2active[j]];
				else
					buffer[j]=(float64_t) kernel(docnum, j);
			}
		}
		else
		{
			for(i=0;(j=active2dnum[i])>=0;i++)
			{
				if(kernel_cache.totdoc2active[j] >= 0)
					buffer[j]=kernel_cache.buffer[start+kernel_cache.totdoc2active[j]];
				else
				{
					int32_t k=j;
					if (k>=num_vectors)
						k=2*num_vectors-1-k;
					buffer[j]=(float64_t) kernel(docnum, k);
				}
			}
		}
	}
	else
	{
		if (full_line)
		{
			for(j=0;j<get_num_vec_lhs();j++)
				buffer[j]=(KERNELCACHE_ELEM) kernel(docnum, j);
		}
		else
		{
			for(i=0;(j=active2dnum[i])>=0;i++)
			{
				int32_t k=j;
				if (k>=num_vectors)
					k=2*num_vectors-1-k;
				buffer[j]=(KERNELCACHE_ELEM) kernel(docnum, k);
			}
		}
	}
}


// Fills cache for the row m
void Kernel::cache_kernel_row(int32_t m)
{
	int32_t j,k,l;
	KERNELCACHE_ELEM *cache;

	int32_t num_vectors = get_num_vec_lhs();

	if (m>=num_vectors)
		m=2*num_vectors-1-m;

	if(!kernel_cache_check(m))   // not cached yet
	{
		cache = kernel_cache_clean_and_malloc(m);
		if(cache) {
			l=kernel_cache.totdoc2active[m];

			for(j=0;j<kernel_cache.activenum;j++)  // fill cache
			{
				k=kernel_cache.active2totdoc[j];

				if((kernel_cache.index[k] != -1) && (l != -1) && (k != m)) {
					cache[j]=kernel_cache.buffer[((KERNELCACHE_IDX) kernel_cache.activenum)
						*kernel_cache.index[k]+l];
				}
				else
				{
					if (k>=num_vectors)
						k=2*num_vectors-1-k;

					cache[j]=kernel(m, k);
				}
			}
		}
		else
			perror("Error: Kernel cache full! => increase cache size");
	}
}


void* Kernel::cache_multiple_kernel_row_helper(void* p)
{
	int32_t j,k,l;
	S_KTHREAD_PARAM* params = (S_KTHREAD_PARAM*) p;

	for (int32_t i=params->start; i<params->end; i++)
	{
		KERNELCACHE_ELEM* cache=params->cache[i];
		int32_t m = params->uncached_rows[i];
		l=params->kernel_cache->totdoc2active[m];

		for(j=0;j<params->kernel_cache->activenum;j++)  // fill cache
		{
			k=params->kernel_cache->active2totdoc[j];

			if((params->kernel_cache->index[k] != -1) && (l != -1) && (!params->needs_computation[k])) {
				cache[j]=params->kernel_cache->buffer[((KERNELCACHE_IDX) params->kernel_cache->activenum)
					*params->kernel_cache->index[k]+l];
			}
			else
				{
					if (k>=params->num_vectors)
						k=2*params->num_vectors-1-k;

					cache[j]=params->kernel->kernel(m, k);
				}
		}

		//now line m is cached
		params->needs_computation[m]=0;
	}
	return NULL;
}

// Fills cache for the rows in key
void Kernel::cache_multiple_kernel_rows(int32_t* rows, int32_t num_rows)
{
	int32_t nthreads=env()->get_num_threads();

	if (nthreads<2)
	{
		for(int32_t i=0;i<num_rows;i++)
			cache_kernel_row(rows[i]);
	}
	else
	{
		// fill up kernel cache
		int32_t* uncached_rows = SG_MALLOC(int32_t, num_rows);
		KERNELCACHE_ELEM** cache = SG_MALLOC(KERNELCACHE_ELEM*, num_rows);
		S_KTHREAD_PARAM params;
		int32_t num_threads=nthreads-1;
		int32_t num_vec=get_num_vec_lhs();
		ASSERT(num_vec>0)
		uint8_t* needs_computation=SG_CALLOC(uint8_t, num_vec);

		int32_t step=0;
		int32_t num=0;
		int32_t end=0;

		// allocate cachelines if necessary
		for (int32_t i=0; i<num_rows; i++)
		{
			int32_t idx=rows[i];
			if (idx>=num_vec)
				idx=2*num_vec-1-idx;

			if (kernel_cache_check(idx))
				continue;

			needs_computation[idx]=1;
			uncached_rows[num]=idx;
			cache[num]= kernel_cache_clean_and_malloc(idx);

			if (!cache[num])
				error("Kernel cache full! => increase cache size");

			num++;
		}

		if (num>0)
		{
			step = num/nthreads;

			if (step<1)
			{
				num_threads=num-1;
				step=1;
			}

			#pragma omp parallel for private(params)
			for (int32_t t=0; t<num_threads; t++)
			{
				params.kernel = this;
				params.kernel_cache = &kernel_cache;
				params.cache = cache;
				params.uncached_rows = uncached_rows;
				params.needs_computation = needs_computation;
				params.num_uncached = num;
				params.start = t*step;
				params.end = (t+1)*step;
				params.num_vectors = get_num_vec_lhs();
				end=params.end;

				cache_multiple_kernel_row_helper(&params);
			}
		}
		else
			num_threads=-1;


		S_KTHREAD_PARAM last_param;
		last_param.kernel = this;
		last_param.kernel_cache = &kernel_cache;
		last_param.cache = cache;
		last_param.uncached_rows = uncached_rows;
		last_param.needs_computation = needs_computation;
		last_param.start = end;
		last_param.num_uncached = num;
		last_param.end = num;
		last_param.num_vectors = get_num_vec_lhs();

		cache_multiple_kernel_row_helper(&last_param);

		SG_FREE(needs_computation);
		SG_FREE(cache);
		SG_FREE(uncached_rows);
	}
}

// remove numshrink columns in the cache
// which correspond to examples marked
void Kernel::kernel_cache_shrink(
	int32_t totdoc, int32_t numshrink, int32_t *after)
{
	ASSERT(totdoc > 0);
	int32_t i,j,jj,scount;     // 0 in after.
	KERNELCACHE_IDX from=0,to=0;
	int32_t *keep;

	keep=SG_MALLOC(int32_t, totdoc);
	for(j=0;j<totdoc;j++) {
		keep[j]=1;
	}
	scount=0;
	for(jj=0;(jj<kernel_cache.activenum) && (scount<numshrink);jj++) {
		j=kernel_cache.active2totdoc[jj];
		if(!after[j]) {
			scount++;
			keep[j]=0;
		}
	}

	for(i=0;i<kernel_cache.max_elems;i++) {
		for(jj=0;jj<kernel_cache.activenum;jj++) {
			j=kernel_cache.active2totdoc[jj];
			if(!keep[j]) {
				from++;
			}
			else {
				kernel_cache.buffer[to]=kernel_cache.buffer[from];
				to++;
				from++;
			}
		}
	}

	kernel_cache.activenum=0;
	for(j=0;j<totdoc;j++) {
		if((keep[j]) && (kernel_cache.totdoc2active[j] != -1)) {
			kernel_cache.active2totdoc[kernel_cache.activenum]=j;
			kernel_cache.totdoc2active[j]=kernel_cache.activenum;
			kernel_cache.activenum++;
		}
		else {
			kernel_cache.totdoc2active[j]=-1;
		}
	}

	kernel_cache.max_elems= (int32_t) kernel_cache.buffsize;

	if (kernel_cache.activenum>0)
		kernel_cache.buffsize/=kernel_cache.activenum;

	if(kernel_cache.max_elems>totdoc)
		kernel_cache.max_elems=totdoc;

	SG_FREE(keep);

}

void Kernel::kernel_cache_reset_lru()
{
	int32_t maxlru=0,k;

	for(k=0;k<kernel_cache.max_elems;k++) {
		if(maxlru < kernel_cache.lru[k])
			maxlru=kernel_cache.lru[k];
	}
	for(k=0;k<kernel_cache.max_elems;k++) {
		kernel_cache.lru[k]-=maxlru;
	}
}

void Kernel::kernel_cache_cleanup()
{
	SG_FREE(kernel_cache.index);
	SG_FREE(kernel_cache.occu);
	SG_FREE(kernel_cache.lru);
	SG_FREE(kernel_cache.invindex);
	SG_FREE(kernel_cache.active2totdoc);
	SG_FREE(kernel_cache.totdoc2active);
	SG_FREE(kernel_cache.buffer);
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
}

int32_t Kernel::kernel_cache_malloc()
{
  int32_t i;

  if(kernel_cache_space_available()) {
    for(i=0;i<kernel_cache.max_elems;i++) {
      if(!kernel_cache.occu[i]) {
	kernel_cache.occu[i]=1;
	kernel_cache.elems++;
	return(i);
      }
    }
  }
  return(-1);
}

void Kernel::kernel_cache_free(int32_t cacheidx)
{
	kernel_cache.occu[cacheidx]=0;
	kernel_cache.elems--;
}

// remove least recently used cache
// element
int32_t Kernel::kernel_cache_free_lru()
{
  int32_t k,least_elem=-1,least_time;

  least_time=kernel_cache.time+1;
  for(k=0;k<kernel_cache.max_elems;k++) {
    if(kernel_cache.invindex[k] != -1) {
      if(kernel_cache.lru[k]<least_time) {
	least_time=kernel_cache.lru[k];
	least_elem=k;
      }
    }
  }

  if(least_elem != -1) {
    kernel_cache_free(least_elem);
    kernel_cache.index[kernel_cache.invindex[least_elem]]=-1;
    kernel_cache.invindex[least_elem]=-1;
    return(1);
  }
  return(0);
}

// Get a free cache entry. In case cache is full, the lru
// element is removed.
KERNELCACHE_ELEM* Kernel::kernel_cache_clean_and_malloc(int32_t cacheidx)
{
	int32_t result;
	if((result = kernel_cache_malloc()) == -1) {
		if(kernel_cache_free_lru()) {
			result = kernel_cache_malloc();
		}
	}
	kernel_cache.index[cacheidx]=result;
	if(result == -1) {
		return(0);
	}
	kernel_cache.invindex[result]=cacheidx;
	kernel_cache.lru[kernel_cache.index[cacheidx]]=kernel_cache.time; // lru
	return &kernel_cache.buffer[((KERNELCACHE_IDX) kernel_cache.activenum)*kernel_cache.index[cacheidx]];
}
#endif //USE_SVMLIGHT

void Kernel::load(std::shared_ptr<File> loader)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
}

void Kernel::save(std::shared_ptr<File> writer)
{
	SGMatrix<float64_t> k_matrix=get_kernel_matrix<float64_t>();
	SG_SET_LOCALE_C;
	writer->set_matrix(k_matrix.matrix, k_matrix.num_rows, k_matrix.num_cols);
	SG_RESET_LOCALE;
}

void Kernel::remove_lhs_and_rhs()
{
	if (rhs!=lhs)

	rhs = NULL;
	num_rhs=0;


	lhs = NULL;
	num_lhs=0;
	lhs_equals_rhs=false;

#ifdef USE_SVMLIGHT
	cache_reset();
#endif //USE_SVMLIGHT
}

void Kernel::remove_lhs()
{
	if (rhs==lhs)
		rhs=NULL;

	lhs = NULL;
	num_lhs=0;
	lhs_equals_rhs=false;
#ifdef USE_SVMLIGHT
	cache_reset();
#endif //USE_SVMLIGHT
}

/// takes all necessary steps if the rhs is removed from kernel
void Kernel::remove_rhs()
{
	if (rhs!=lhs)

	rhs = NULL;
	num_rhs=0;
	lhs_equals_rhs=false;

#ifdef USE_SVMLIGHT
	cache_reset();
#endif //USE_SVMLIGHT
}

#define ENUM_CASE(n) case n: io::info(#n " "); break;

void Kernel::list_kernel()
{
	io::info("{} - \"{}\" weight={:1.2f} OPT:{}", fmt::ptr(this), get_name(),
			get_combined_kernel_weight(),
			get_optimization_type()==FASTBUTMEMHUNGRY ? "FASTBUTMEMHUNGRY" :
			"SLOWBUTMEMEFFICIENT");

	switch (get_kernel_type())
	{
		ENUM_CASE(K_UNKNOWN)
		ENUM_CASE(K_LINEAR)
		ENUM_CASE(K_POLY)
		ENUM_CASE(K_GAUSSIAN)
		ENUM_CASE(K_GAUSSIANSHIFT)
		ENUM_CASE(K_GAUSSIANMATCH)
		ENUM_CASE(K_GAUSSIANCOMPACT)
		ENUM_CASE(K_HISTOGRAM)
		ENUM_CASE(K_SALZBERG)
		ENUM_CASE(K_LOCALITYIMPROVED)
		ENUM_CASE(K_SIMPLELOCALITYIMPROVED)
		ENUM_CASE(K_FIXEDDEGREE)
		ENUM_CASE(K_WEIGHTEDDEGREE)
		ENUM_CASE(K_WEIGHTEDDEGREEPOS)
		ENUM_CASE(K_WEIGHTEDDEGREERBF)
		ENUM_CASE(K_WEIGHTEDCOMMWORDSTRING)
		ENUM_CASE(K_POLYMATCH)
		ENUM_CASE(K_ALIGNMENT)
		ENUM_CASE(K_COMMWORDSTRING)
		ENUM_CASE(K_COMMULONGSTRING)
		ENUM_CASE(K_SPECTRUMRBF)
		ENUM_CASE(K_COMBINED)
		ENUM_CASE(K_AUC)
		ENUM_CASE(K_CUSTOM)
		ENUM_CASE(K_SIGMOID)
		ENUM_CASE(K_CHI2)
		ENUM_CASE(K_DIAG)
		ENUM_CASE(K_CONST)
		ENUM_CASE(K_DISTANCE)
		ENUM_CASE(K_LOCALALIGNMENT)
		ENUM_CASE(K_PYRAMIDCHI2)
		ENUM_CASE(K_OLIGO)
		ENUM_CASE(K_MATCHWORD)
		ENUM_CASE(K_TPPK)
		ENUM_CASE(K_REGULATORYMODULES)
		ENUM_CASE(K_SPARSESPATIALSAMPLE)
		ENUM_CASE(K_HISTOGRAMINTERSECTION)
		ENUM_CASE(K_WAVELET)
		ENUM_CASE(K_WAVE)
		ENUM_CASE(K_CAUCHY)
		ENUM_CASE(K_TSTUDENT)
		ENUM_CASE(K_MULTIQUADRIC)
		ENUM_CASE(K_EXPONENTIAL)
		ENUM_CASE(K_RATIONAL_QUADRATIC)
		ENUM_CASE(K_POWER)
		ENUM_CASE(K_SPHERICAL)
		ENUM_CASE(K_LOG)
		ENUM_CASE(K_SPLINE)
		ENUM_CASE(K_ANOVA)
		ENUM_CASE(K_CIRCULAR)
		ENUM_CASE(K_INVERSEMULTIQUADRIC)
		ENUM_CASE(K_SPECTRUMMISMATCHRBF)
		ENUM_CASE(K_DISTANTSEGMENTS)
		ENUM_CASE(K_BESSEL)
		ENUM_CASE(K_JENSENSHANNON)
		ENUM_CASE(K_DIRECTOR)
		ENUM_CASE(K_PRODUCT)
		ENUM_CASE(K_EXPONENTIALARD)
		ENUM_CASE(K_GAUSSIANARD)
		ENUM_CASE(K_GAUSSIANARDSPARSE)
		ENUM_CASE(K_STREAMING)
		ENUM_CASE(K_PERIODIC)
	}

	switch (get_feature_class())
	{
		ENUM_CASE(C_UNKNOWN)
		ENUM_CASE(C_DENSE)
		ENUM_CASE(C_SPARSE)
		ENUM_CASE(C_STRING)
		ENUM_CASE(C_STREAMING_DENSE)
		ENUM_CASE(C_STREAMING_SPARSE)
		ENUM_CASE(C_STREAMING_STRING)
		ENUM_CASE(C_STREAMING_VW)
		ENUM_CASE(C_COMBINED)
		ENUM_CASE(C_COMBINED_DOT)
		ENUM_CASE(C_WD)
		ENUM_CASE(C_SPEC)
		ENUM_CASE(C_WEIGHTEDSPEC)
		ENUM_CASE(C_POLY)
		ENUM_CASE(C_BINNED_DOT)
		ENUM_CASE(C_DIRECTOR_DOT)
		ENUM_CASE(C_LATENT)
		ENUM_CASE(C_MATRIX)
		ENUM_CASE(C_FACTOR_GRAPH)
		ENUM_CASE(C_INDEX)
		ENUM_CASE(C_SUB_SAMPLES_DENSE)
		ENUM_CASE(C_ANY)
	}

	switch (get_feature_type())
	{
		ENUM_CASE(F_UNKNOWN)
		ENUM_CASE(F_BOOL)
		ENUM_CASE(F_CHAR)
		ENUM_CASE(F_BYTE)
		ENUM_CASE(F_SHORT)
		ENUM_CASE(F_WORD)
		ENUM_CASE(F_INT)
		ENUM_CASE(F_UINT)
		ENUM_CASE(F_LONG)
		ENUM_CASE(F_ULONG)
		ENUM_CASE(F_SHORTREAL)
		ENUM_CASE(F_DREAL)
		ENUM_CASE(F_LONGREAL)
		ENUM_CASE(F_ANY)
	}
	io::info("");
}
#undef ENUM_CASE

bool Kernel::init_optimization(
	int32_t count, int32_t *IDX, float64_t * weights)
{
   error("kernel does not support linadd optimization");
	return false ;
}

bool Kernel::delete_optimization()
{
   error("kernel does not support linadd optimization");
	return false;
}

float64_t Kernel::compute_optimized(int32_t vector_idx)
{
   error("kernel does not support linadd optimization");
	return 0;
}

void Kernel::compute_batch(
	int32_t num_vec, int32_t* vec_idx, float64_t* target, int32_t num_suppvec,
	int32_t* IDX, float64_t* weights, float64_t factor)
{
   error("kernel does not support batch computation");
}

void Kernel::add_to_normal(int32_t vector_idx, float64_t weight)
{
   error("kernel does not support linadd optimization, add_to_normal not implemented");
}

void Kernel::clear_normal()
{
   error("kernel does not support linadd optimization, clear_normal not implemented");
}

int32_t Kernel::get_num_subkernels()
{
	return 1;
}

void Kernel::compute_by_subkernel(
	int32_t vector_idx, float64_t * subkernel_contrib)
{
   error("kernel compute_by_subkernel not implemented");
}

const float64_t* Kernel::get_subkernel_weights(int32_t &num_weights)
{
	num_weights=1 ;
	return &combined_kernel_weight ;
}

SGVector<float64_t> Kernel::get_subkernel_weights()
{
	int num_weights = 1;
	const float64_t* weight = get_subkernel_weights(num_weights);
	return SGVector<float64_t>(const_cast<float64_t*>(weight),1,false);
}

void Kernel::set_subkernel_weights(const SGVector<float64_t> weights)
{
	ASSERT(weights.vector)
	if (weights.vlen!=1)
      error("number of subkernel weights should be one ...");

	combined_kernel_weight = weights.vector[0] ;
}

std::shared_ptr<Kernel> Kernel::obtain_from_generic(std::shared_ptr<SGObject> kernel)
{
	if (kernel)
	{
		auto casted=std::dynamic_pointer_cast<Kernel>(kernel);
		require(casted, "Kernel::obtain_from_generic(): Error, provided object"
				" of class \"{}\" is not a subclass of Kernel!",
				kernel->get_name());
		return casted;
	}
	else
		return NULL;
}

bool Kernel::init_optimization_svm(std::shared_ptr<SVM > svm)
{
	int32_t num_suppvec=svm->get_num_support_vectors();
	int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
	float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);

	for (int32_t i=0; i<num_suppvec; i++)
	{
		sv_idx[i]    = svm->get_support_vector(i);
		sv_weight[i] = svm->get_alpha(i);
	}
	bool ret = init_optimization(num_suppvec, sv_idx, sv_weight);

	SG_FREE(sv_idx);
	SG_FREE(sv_weight);
	return ret;
}

void Kernel::load_serializable_post() noexcept(false)
{
	SGObject::load_serializable_post();
	if (lhs_equals_rhs)
		rhs=lhs;
}

void Kernel::save_serializable_pre() noexcept(false)
{
	SGObject::save_serializable_pre();

	if (lhs_equals_rhs)
		rhs=NULL;
}

void Kernel::save_serializable_post() noexcept(false)
{
	SGObject::save_serializable_post();

	if (lhs_equals_rhs)
		rhs=lhs;
}

void Kernel::register_params()
{
	SG_ADD(&cache_size, "cache_size", "Cache size in MB.");
	SG_ADD(
		&lhs, "lhs", "Feature vectors to occur on left hand side.",
		ParameterProperties::READONLY);
	SG_ADD(
		&rhs, "rhs", "Feature vectors to occur on right hand side.",
		ParameterProperties::READONLY);
	SG_ADD(&lhs_equals_rhs, "lhs_equals_rhs",
		"If features on lhs are the same as on rhs.");
	SG_ADD(&num_lhs, "num_lhs", "Number of feature vectors on left hand side.");
	SG_ADD(
	    &num_rhs, "num_rhs", "Number of feature vectors on right hand side.");
	SG_ADD(
	    &combined_kernel_weight, "combined_kernel_weight",
	    "Combined kernel weight.", ParameterProperties::HYPER);
	SG_ADD(
	    &optimization_initialized, "optimization_initialized",
	    "Optimization is initialized.");
	SG_ADD(&properties, "properties", "Kernel properties.");
	SG_ADD(
	    &normalizer, "normalizer", "Normalize the kernel.",
	    ParameterProperties::HYPER);

	SG_ADD_OPTIONS(
	    (machine_int_t*)&opt_type, "opt_type", "Optimization type.",
	    ParameterProperties::NONE,
	    SG_OPTIONS(FASTBUTMEMHUNGRY, SLOWBUTMEMEFFICIENT));
}


void Kernel::init()
{
	cache_size=10;
	kernel_matrix=NULL;
	lhs=NULL;
	rhs=NULL;
	num_lhs=0;
	num_rhs=0;
	lhs_equals_rhs=false;
	combined_kernel_weight=1;
	optimization_initialized=false;
	opt_type=FASTBUTMEMHUNGRY;
	properties=KP_NONE;
	normalizer=NULL;

#ifdef USE_SVMLIGHT
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
#endif //USE_SVMLIGHT

	set_normalizer(std::make_shared<IdentityKernelNormalizer>());
}

namespace shogun
{
/** kernel thread parameters */
template <class T> struct K_THREAD_PARAM
{
	/** kernel */
	Kernel* kernel;
	/** start (unit row) */
	int32_t start;
	/** end (unit row) */
	int32_t end;
	/** start (unit number of elements) */
	int64_t total_start;
	/** m */
	int32_t m;
	/** n */
	int32_t n;
	/** result */
	T* result;
	/** kernel matrix k(i,j)=k(j,i) */
	bool symmetric;
	/** output progress */
	bool verbose;
	/* Progress bar*/
	PRange<int64_t>* pb;
};
}

float64_t Kernel::sum_symmetric_block(index_t block_begin, index_t block_size,
		bool no_diag)
{
	SG_TRACE("Entering");

	require(has_features(), "No features assigned to kernel");
	require(lhs_equals_rhs, "The kernel matrix is not symmetric!");
	require(block_begin>=0 && block_begin<num_rhs,
			"Invalid block begin index ({}, {})!", block_begin, block_begin);
	require(block_begin+block_size<=num_rhs,
			"Invalid block size ({}) at starting index ({}, {})! "
			"Please use smaller blocks!", block_size, block_begin, block_begin);
	require(block_size>=1, "Invalid block size ({})!", block_size);

	float64_t sum=0.0;

	// since the block is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part.
	// this can be done in parallel
	#pragma omp parallel for reduction(+:sum)
	for (index_t i=0; i<block_size; ++i)
	{
		// compute the kernel values on the upper triangular part of the kernel
		// matrix and compute sum on the fly
		for (index_t j=i+1; j<block_size; ++j)
		{
			float64_t k=kernel(i+block_begin, j+block_begin);
			sum+=k;
		}
	}

	// the actual sum would be twice of what we computed
	sum*=2;

	// add the diagonal elements if required - keeping this check
	// outside of the loop to save cycles
	if (!no_diag)
	{
		#pragma omp parallel for reduction(+:sum)
		for (index_t i=0; i<block_size; ++i)
		{
			float64_t diag=kernel(i+block_begin, i+block_begin);
			sum+=diag;
		}
	}

	SG_TRACE("Leaving");

	return sum;
}

float64_t Kernel::sum_block(index_t block_begin_row, index_t block_begin_col,
		index_t block_size_row, index_t block_size_col, bool no_diag)
{
	SG_TRACE("Entering");

	require(has_features(), "No features assigned to kernel");
	require(block_begin_row>=0 && block_begin_row<num_lhs &&
			block_begin_col>=0 && block_begin_col<num_rhs,
			"Invalid block begin index ({}, {})!",
			block_begin_row, block_begin_col);
	require(block_begin_row+block_size_row<=num_lhs &&
			block_begin_col+block_size_col<=num_rhs,
			"Invalid block size ({}, {}) at starting index ({}, {})! "
			"Please use smaller blocks!", block_size_row, block_size_col,
			block_begin_row, block_begin_col);
	require(block_size_row>=1 && block_size_col>=1,
			"Invalid block size ({}, {})!", block_size_row, block_size_col);

	// check if removal of diagonal is required/valid
	if (no_diag && block_size_row!=block_size_col)
	{
		io::warn("Not removing the main diagonal since block is not square!");
		no_diag=false;
	}

	float64_t sum=0.0;

	// this can be done in parallel for the rows/cols
	#pragma omp parallel for reduction(+:sum)
	for (index_t i=0; i<block_size_row; ++i)
	{
		// compute the kernel values and compute sum on the fly
		for (index_t j=0; j<block_size_col; ++j)
		{
			float64_t k=no_diag && i==j ? 0 :
				kernel(i+block_begin_row, j+block_begin_col);
			sum+=k;
		}
	}

	SG_TRACE("Leaving");

	return sum;
}

SGVector<float64_t> Kernel::row_wise_sum_symmetric_block(index_t block_begin,
		index_t block_size, bool no_diag)
{
	SG_TRACE("Entering");

	require(has_features(), "No features assigned to kernel");
	require(lhs_equals_rhs, "The kernel matrix is not symmetric!");
	require(block_begin>=0 && block_begin<num_rhs,
			"Invalid block begin index ({}, {})!", block_begin, block_begin);
	require(block_begin+block_size<=num_rhs,
			"Invalid block size ({}) at starting index ({}, {})! "
			"Please use smaller blocks!", block_size, block_begin, block_begin);
	require(block_size>=1, "Invalid block size ({})!", block_size);

	// initialize the vector that accumulates the row/col-wise sum on the go
	SGVector<float64_t> row_sum(block_size);
	row_sum.set_const(0.0);

	// since the block is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part.
	// this can be done in parallel for the rows/cols
	#pragma omp parallel for
	for (index_t i=0; i<block_size; ++i)
	{
		// compute the kernel values on the upper triangular part of the kernel
		// matrix and compute row-wise sum on the fly
		for (index_t j=i+1; j<block_size; ++j)
		{
			float64_t k=kernel(i+block_begin, j+block_begin);
			#pragma omp critical
			{
				row_sum[i]+=k;
				row_sum[j]+=k;
			}
		}
	}

	// add the diagonal elements if required - keeping this check
	// outside of the loop to save cycles
	if (!no_diag)
	{
		#pragma omp parallel for
		for (index_t i=0; i<block_size; ++i)
		{
			float64_t diag=kernel(i+block_begin, i+block_begin);
			row_sum[i]+=diag;
		}
	}

	SG_TRACE("Leaving");

	return row_sum;
}

SGMatrix<float64_t> Kernel::row_wise_sum_squared_sum_symmetric_block(index_t
		block_begin, index_t block_size, bool no_diag)
{
	SG_TRACE("Entering");

	require(has_features(), "No features assigned to kernel");
	require(lhs_equals_rhs, "The kernel matrix is not symmetric!");
	require(block_begin>=0 && block_begin<num_rhs,
			"Invalid block begin index ({}, {})!", block_begin, block_begin);
	require(block_begin+block_size<=num_rhs,
			"Invalid block size ({}) at starting index ({}, {})! "
			"Please use smaller blocks!", block_size, block_begin, block_begin);
	require(block_size>=1, "Invalid block size ({})!", block_size);

	// initialize the matrix that accumulates the row/col-wise sum on the go
	// the first column stores the sum of kernel values
	// the second column stores the sum of squared kernel values
	SGMatrix<float64_t> row_sum(block_size, 2);
	row_sum.set_const(0.0);

	// since the block is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part
	// this can be done in parallel for the rows/cols
#pragma omp parallel for
	for (index_t i=0; i<block_size; ++i)
	{
		// compute the kernel values on the upper triangular part of the kernel
		// matrix and compute row-wise sum and squared sum on the fly
		for (index_t j=i+1; j<block_size; ++j)
		{
			float64_t k=kernel(i+block_begin, j+block_begin);
#pragma omp critical
			{
				row_sum(i, 0)+=k;
				row_sum(j, 0)+=k;
				row_sum(i, 1)+=k*k;
				row_sum(j, 1)+=k*k;
			}
		}
	}

	// add the diagonal elements if required - keeping this check
	// outside of the loop to save cycles
	if (!no_diag)
	{
#pragma omp parallel for
		for (index_t i=0; i<block_size; ++i)
		{
			float64_t diag=kernel(i+block_begin, i+block_begin);
			row_sum(i, 0)+=diag;
			row_sum(i, 1)+=diag*diag;
		}
	}

	SG_TRACE("Leaving");

	return row_sum;
}

SGVector<float64_t> Kernel::row_col_wise_sum_block(index_t block_begin_row,
		index_t block_begin_col, index_t block_size_row,
		index_t block_size_col, bool no_diag)
{
	SG_TRACE("Entering");

	require(has_features(), "No features assigned to kernel");
	require(block_begin_row>=0 && block_begin_row<num_lhs &&
			block_begin_col>=0 && block_begin_col<num_rhs,
			"Invalid block begin index ({}, {})!",
			block_begin_row, block_begin_col);
	require(block_begin_row+block_size_row<=num_lhs &&
			block_begin_col+block_size_col<=num_rhs,
			"Invalid block size ({}, {}) at starting index ({}, {})! "
			"Please use smaller blocks!", block_size_row, block_size_col,
			block_begin_row, block_begin_col);
	require(block_size_row>=1 && block_size_col>=1,
			"Invalid block size ({}, {})!", block_size_row, block_size_col);

	// check if removal of diagonal is required/valid
	if (no_diag && block_size_row!=block_size_col)
	{
		io::warn("Not removing the main diagonal since block is not square!");
		no_diag=false;
	}

	// initialize the vector that accumulates the row/col-wise sum on the go
	// the first block_size_row entries store the row-wise sum of kernel values
	// the nextt block_size_col entries store the col-wise sum of kernel values
	SGVector<float64_t> sum(block_size_row+block_size_col);
	sum.set_const(0.0);

	// this can be done in parallel for the rows/cols
#pragma omp parallel for
	for (index_t i=0; i<block_size_row; ++i)
	{
		// compute the kernel values and compute sum on the fly
		for (index_t j=0; j<block_size_col; ++j)
		{
			float64_t k=no_diag && i==j ? 0 :
				kernel(i+block_begin_row, j+block_begin_col);
#pragma omp critical
			{
				sum[i]+=k;
				sum[j+block_size_row]+=k;
			}
		}
	}

	SG_TRACE("Leaving");

	return sum;
}

template <class T> void* Kernel::get_kernel_matrix_helper(void* p)
{
	K_THREAD_PARAM<T>* params= (K_THREAD_PARAM<T>*) p;
	int32_t i_start=params->start;
	int32_t i_end=params->end;
	Kernel* k=params->kernel;
	T* result=params->result;
	bool symmetric=params->symmetric;
	int32_t n=params->n;
	int32_t m=params->m;
	bool verbose=params->verbose;
	int64_t total_start=params->total_start;
	int64_t total=total_start;
	PRange<int64_t>* pb = params->pb;

	for (int32_t i=i_start; i<i_end; i++)
	{
		int32_t j_start=0;

		if (symmetric)
			j_start=i;

		for (int32_t j=j_start; j<n; j++)
		{
			float64_t v=k->kernel(i,j);
			result[i+j*m]=v;

			if (symmetric && i!=j)
				result[j+i*m]=v;

			if (verbose)
			{
				total++;

				if (symmetric && i!=j)
					total++;

				pb->print_progress();

				// TODO: replace with the new signal
				// if (Signal::cancel_computations())
				//	break;
			}
		}

	}

	return NULL;
}

template <class T>
SGMatrix<T> Kernel::get_kernel_matrix()
{
	T* result = NULL;

	require(has_features(), "no features assigned to kernel");

	int32_t m=get_num_vec_lhs();
	int32_t n=get_num_vec_rhs();

	int64_t total_num = int64_t(m)*n;

	// if lhs == rhs and sizes match assume k(i,j)=k(j,i)
	bool symmetric= (lhs && lhs==rhs && m==n);

	SG_DEBUG("returning kernel matrix of size {}x{}", m, n)

	result=SG_MALLOC(T, total_num);

	int32_t num_threads=env()->get_num_threads();
	K_THREAD_PARAM<T> params;
	int64_t step = total_num/num_threads;
	index_t t = 0;
	auto pb = SG_PROGRESS(range(total_num));
#pragma omp parallel for lastprivate(t) private(params)
	for (t = 0; t < num_threads; ++t)
	{
		params.kernel = this;
		params.result = result;
		params.start = compute_row_start(t*step, n, symmetric);
		params.end = compute_row_start((t+1)*step, n, symmetric);
		params.total_start=t*step;
		params.n=n;
		params.m=m;
		params.symmetric=symmetric;
		params.verbose=false;
		params.pb = &pb;
		Kernel::get_kernel_matrix_helper<T>((void*)&params);
	}

	if (total_num % num_threads != 0)
	{
		params.kernel = this;
		params.result = result;
		params.start = compute_row_start(t*step, n, symmetric);
		params.end = m;
		params.total_start=t*step;
		params.n=n;
		params.m=m;
		params.symmetric=symmetric;
		params.verbose=false;
		params.pb = &pb;
		Kernel::get_kernel_matrix_helper<T>((void*)&params);
	}

	pb.complete();

	return SGMatrix<T>(result,m,n,true);
}


template SGMatrix<float64_t> Kernel::get_kernel_matrix<float64_t>();
template SGMatrix<float32_t> Kernel::get_kernel_matrix<float32_t>();

template void* Kernel::get_kernel_matrix_helper<float64_t>(void* p);
template void* Kernel::get_kernel_matrix_helper<float32_t>(void* p);
