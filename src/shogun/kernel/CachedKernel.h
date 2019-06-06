#ifndef CACHED_KERNEL
#define CACHED_KERNEL

#include <shogun/kernel/Kernel.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

namespace shogun
{

#ifdef USE_SVMLIGHT
#ifndef SWIG

#ifdef USE_SHORTREAL_KERNELCACHE
	/** kernel cache element */
	typedef float32_t KERNELCACHE_ELEM;
#else
	/** kernel cache element */
	typedef float64_t KERNELCACHE_ELEM;
#endif // USE_SHORTREAL_KERNELCACHE

	/** kernel cache index */
	typedef int64_t KERNELCACHE_IDX;

	class CachedKernel
	{
	public:
		CachedKernel(CKernel* kernel = nullptr);

		CachedKernel(CachedKernel&& cached_kernel);

		SG_DELETE_COPY_AND_ASSIGN(CachedKernel);

		/** destructor */
		~CachedKernel();

		/** operator overloading for easy kernel access */
		CKernel* operator->()
		{
			return kernel;
		}

		void set_kernel(CKernel* kernel);

		CKernel* get_kernel() const
		{
			return kernel;
		}

		/** cache reset */
		inline void cache_reset()
		{
			resize_kernel_cache(kernel->get_cache_size());
		}

		/** get maximum elements in cache
		 *
		 * @return maximum elements in cache
		 */
		inline int32_t get_max_elems_cache()
		{
			return kernel_cache.max_elems;
		}

		/** get activenum cache
		 *
		 * @return activecnum cache
		 */
		inline int32_t get_activenum_cache()
		{
			return kernel_cache.activenum;
		}

		/** get kernel row
		 *
		 * @param docnum docnum
		 * @param active2dnum active2dnum
		 * @param buffer buffer
		 * @param full_line full line
		 */
		void get_kernel_row(
		    int32_t docnum, int32_t* active2dnum, float64_t* buffer,
		    bool full_line = false);

		/** cache kernel row
		 *
		 * @param x x
		 */
		void cache_kernel_row(int32_t x);

		/** cache multiple kernel rows
		 *
		 * @param key key
		 * @param varnum
		 */
		void cache_multiple_kernel_rows(int32_t* key, int32_t varnum);

		/** kernel cache reset lru */
		void kernel_cache_reset_lru();

		/** kernel cache shrink
		 *
		 * @param totdoc totdoc
		 * @param num_shrink number of shrink
		 * @param after after
		 */
		void
		kernel_cache_shrink(int32_t totdoc, int32_t num_shrink, int32_t* after);

		/** resize kernel cache
		 *
		 * @param size new size
		 * @param regression_hack hack for regression
		 */
		void
		resize_kernel_cache(KERNELCACHE_IDX size, bool regression_hack = false);

		/** set the lru time
		 *
		 * @param t the time to use
		 */
		inline void set_time(int32_t t)
		{
			kernel_cache.time = t;
		}

		/** update lru time of item at given index to avoid removal from cache
		 *
		 * @param cacheidx index in cache
		 * @return if updating was successful
		 */
		inline int32_t kernel_cache_touch(int32_t cacheidx)
		{
			if (kernel_cache.index[cacheidx] != -1)
			{
				kernel_cache.lru[kernel_cache.index[cacheidx]] =
				    kernel_cache.time;
				return (1);
			}
			return (0);
		}

		/** check if row at given index is cached
		 *
		 * @param cacheidx index in cache
		 * @return if row at given index is cached
		 */
		inline int32_t kernel_cache_check(int32_t cacheidx)
		{
			return (kernel_cache.index[cacheidx] >= 0);
		}

		/** check if there is room for one more row in kernel cache
		 *
		 * @return if there is room for one more row in kernel cache
		 */
		inline int32_t kernel_cache_space_available()
		{
			return (kernel_cache.elems < kernel_cache.max_elems);
		}

		/** initialize kernel cache
		 *
		 * @param size size to initialize to
		 * @param regression_hack if hack for regression shall be applied
		 */
		void kernel_cache_init(int32_t size, bool regression_hack = false);

		/** cleanup kernel cache */
		void kernel_cache_cleanup();

	private:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/**@ cache kernel evalutations to improve speed */
		struct KERNEL_CACHE
		{
			/** index */
			int32_t* index;
			/** inverse index */
			int32_t* invindex;
			/** active2totdoc */
			int32_t* active2totdoc;
			/** totdoc2active */
			int32_t* totdoc2active;
			/** least recently used */
			int32_t* lru;
			/** occu */
			int32_t* occu;
			/** elements */
			int32_t elems;
			/** max elements */
			int32_t max_elems;
			/** time */
			int32_t time;
			/** active num */
			int32_t activenum;

			/** buffer */
			KERNELCACHE_ELEM* buffer;
			/** buffer size */
			KERNELCACHE_IDX buffsize;
		};

		/** kernel thread parameters */
		struct S_KTHREAD_PARAM
		{
			/** kernel */
			CKernel* kernel;
			/** kernel cache */
			KERNEL_CACHE* kernel_cache;
			/** cache */
			KERNELCACHE_ELEM** cache;
			/** uncached rows */
			int32_t* uncached_rows;
			/** number of uncached rows */
			int32_t num_uncached;
			/** needs computation */
			uint8_t* needs_computation;
			/** start */
			int32_t start;
			/** end */
			int32_t end;
			/** of vectors */
			int32_t num_vectors;
		};
#endif // DOXYGEN_SHOULD_SKIP_THIS

		static void* cache_multiple_kernel_row_helper(void* p);

		void kernel_cache_free(int32_t cacheidx);
		int32_t kernel_cache_malloc();
		int32_t kernel_cache_free_lru();
		KERNELCACHE_ELEM* kernel_cache_clean_and_malloc(int32_t cacheidx);

		/// kernel cache
		KERNEL_CACHE kernel_cache;

		CKernel* kernel;
	};

#endif // SWIG
#endif // USE_SVMLIGHT
} // namespace shogun

#endif // CACHED_KERNEL