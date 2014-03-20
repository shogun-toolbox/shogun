#include <shogun/optimization/framework/CP.h>
using namespace shogun;

CP::CP()
{
	maxCPs = 0;
	ICPcounter = NULL;
	ICPs = NULL;
	ACPs = NULL;
	H_buff = NULL;
	head = NULL;
	map = NULL;
	tail = NULL;
	dim = 0;
	cplist = NULL;
	BufSize = 0;
}

int CP::init(uint32_t mCP, 
			 uint32_t dims,
			 uint32_t bsize)
{
	maxCPs = mCP;
	dim = dims;
	BufSize = bsize;
	ICPcounter= (uint32_t*) BMRM_CALLOC(BufSize, uint32_t);
	ICPs= (float64_t**) BMRM_CALLOC(BufSize, float64_t*);
	ACPs= (uint32_t*) BMRM_CALLOC(BufSize, uint32_t);
	head = (bmrm_ll**) BMRM_CALLOC(1, bmrm_ll*);
	tail = (bmrm_ll**) BMRM_CALLOC(1, bmrm_ll*);
	map = (bool*) BMRM_CALLOC(BufSize, bool);
	cplist = (bmrm_ll*) BMRM_CALLOC(1, bmrm_ll);
	if(ICPcounter == NULL || cplist == NULL || 
	   map == NULL || ICPs == NULL || ACPs == NULL || head == NULL || tail == NULL)
	{
		return -1;
	}
	else
	{
		memset( (bool*) map, true, BufSize);
		return 0;
	}
}

void CP::add_cutting_plane(
		float64_t*	A,
		uint32_t	free_idx,
		float64_t*	cp_data)
{
	REQUIRE(map[free_idx],
		"add_cutting_plane: CP index %u is not free\n", free_idx)

	BMRM_MEMCPY(A+free_idx*(dim), cp_data, (dim)*sizeof(float64_t));
	map[free_idx]=false;

	bmrm_ll *cp=(bmrm_ll*)BMRM_CALLOC(1, bmrm_ll);

	if (cp==NULL)
	{
		SG_SERROR("Out of memory.\n");
		return;
	}

	cp->prev = *tail;
	cp->next = NULL;
	cp->address = A + (free_idx*dim);
	cp->idx = free_idx;
	(*tail)->next=cp;
	*tail=cp;
}

void CP::remove_cutting_plane(
		uint32_t	idxCP)
{
	bmrm_ll *cp_list_ptr=*(head);
	while(cp_list_ptr->address != ICPs[idxCP])
	{
		cp_list_ptr=cp_list_ptr->next;
	}

	if (cp_list_ptr==*(head))
	{
		*head=((*head)->next);
		cp_list_ptr->next->prev=NULL;
	}
	else if (cp_list_ptr==*(tail))
	{
		*tail=(*tail)->prev;
		cp_list_ptr->prev->next=NULL;
	}
	else
	{
		cp_list_ptr->prev->next=cp_list_ptr->next;
		cp_list_ptr->next->prev=cp_list_ptr->prev;
	}

	map[cp_list_ptr->idx]=true;
	BMRM_FREE(cp_list_ptr);
}

void CP::clean_icp(
		BmrmStatistics& bmrm,
		float64_t*& Hmat,
		float64_t*& diag_H,
		float64_t*& beta,
		uint32_t cleanAfter,
		float64_t*& b,
		uint32_t*& I,
		uint32_t cp_models
		)
{
	/* find ICP */
	uint32_t cntICP=0;
	uint32_t cntACP=0;
	bmrm_ll* cp_ptr=*(head);
	uint32_t tmp_idx=0;

	while (cp_ptr != *(tail))
	{
		if (ICPcounter[tmp_idx++]>=cleanAfter)
		{
			ICPs[cntICP++]=cp_ptr->address;
		}
		else
		{
			ACPs[cntACP++]=tmp_idx-1;
		}

		cp_ptr=cp_ptr->next;
	}

	/* do ICP removal */
	if (cntICP > 0)
	{
		uint32_t nCP_new=bmrm.nCP-cntICP;

		for (uint32_t i=0; i<cntICP; ++i)
		{
			tmp_idx=0;
			cp_ptr=*head;

			while(cp_ptr->address != ICPs[i])
			{
				cp_ptr=cp_ptr->next;
				tmp_idx++;
			}

			remove_cutting_plane(i);

			BMRM_MEMMOVE(b+tmp_idx, b+tmp_idx+1,
					(bmrm.nCP+cp_models-tmp_idx)*sizeof(float64_t));
			BMRM_MEMMOVE(beta+tmp_idx, beta+tmp_idx+1,
					(bmrm.nCP-tmp_idx)*sizeof(float64_t));
			BMRM_MEMMOVE(diag_H+tmp_idx, diag_H+tmp_idx+1,
					(bmrm.nCP-tmp_idx)*sizeof(float64_t));
			BMRM_MEMMOVE(I+tmp_idx, I+tmp_idx+1,
					(bmrm.nCP-tmp_idx)*sizeof(uint32_t));
			BMRM_MEMMOVE(ICPcounter+tmp_idx, ICPcounter+tmp_idx+1,
					(bmrm.nCP-tmp_idx)*sizeof(uint32_t));
		}

		/* H */
		for (uint32_t i=0; i < nCP_new; ++i)
		{
			for (uint32_t j=0; j < nCP_new; ++j)
			{
				H_buff[BMRM_INDEX(i, j, maxCPs)]=
					Hmat[BMRM_INDEX(ACPs[i], ACPs[j], maxCPs)];
			}
		}

		for (uint32_t i=0; i<nCP_new; ++i)
			for (uint32_t j=0; j<nCP_new; ++j)
				Hmat[BMRM_INDEX(i, j, maxCPs)]=
					H_buff[BMRM_INDEX(i, j, maxCPs)];

		bmrm.nCP=nCP_new;
		ASSERT(bmrm.nCP<BufSize); 
	}
}

void CP::cleanup()
{
	BMRM_FREE(ICPcounter);
	BMRM_FREE(ACPs);
	BMRM_FREE(ICPs);
	BMRM_FREE(tail);
	BMRM_FREE(head);
	BMRM_FREE(map);
	BMRM_FREE(cplist);
}

