#ifndef CP_H
#define CP_H
#include <shogun/optimization/optdefines.h>
/*Include appropriate headers here*/
namespace shogun
{
struct bmrm_ll
{
	bmrm_ll* prev;
	bmrm_ll* next;
	float64_t* address;
	uint32_t idx;
};

struct CP /* Cutting plane */
{
public:
	/* Inactive cutting planes data */
	uint32_t maxCPs;
	uint32_t* ICPcounter;
	float64_t** ICPs;            
	uint32_t* ACPs;
	float64_t* H_buff;
	bool* map;
	uint32_t BufSize;
	/* Cutting Planes data */
	bmrm_ll** head;
	bmrm_ll** tail;
	bmrm_ll* cplist;
	uint32_t dim;
	
	CP();
	int init(uint32_t mCP, uint32_t dims, uint32_t bsize);
	
	void add_cutting_plane(
		float64_t*	A,
		uint32_t	free_idx,
		float64_t*	cp_data);
		
	void remove_cutting_plane(
		uint32_t idx);

	void clean_icp(
		BmrmStatistics& bmrm,
		float64_t*& H,
		float64_t*& diag_H,
		float64_t*& beta,
		uint32_t cleanAfter,
		float64_t*& b,
		uint32_t*& I,
		uint32_t cp_models = 1
		);

	float64_t * get_cutting_plane(bmrm_ll *ptr) { return ptr->address; }
	
	uint32_t find_free_idx(bool *map, uint32_t size)
	{
	    for (uint32_t i=0; i<size; ++i) if (map[i]) return i;
	    SG_SERROR("No free index available in CP buffer of size %d.\n", size);
	    return size-1;
	}
	
	void cleanup();
};
}

#endif
