#include <stdio.h>
#include <string.h>

#include "lib/Mathematics.h"
#include "lib/config.h"
#include "lib/io.h"
#include "structure/SegmentLoss.h"
#include "lib/Array.h"
#include "lib/Array2.h"
#include "lib/Array3.h"
#include "base/SGObject.h" 
//# define DEBUG

CSegmentLoss::CSegmentLoss()
	:CSGObject(),
	m_segment_loss_matrix(1,1),
	m_segment_loss(1,1,2),
	m_segment_ids(NULL),
	m_segment_mask(NULL),
	m_num_segment_types(NULL)
{
}
CSegmentLoss::~CSegmentLoss()
{
}

void CSegmentLoss::set_segment_loss(float64_t* segment_loss, int32_t m, int32_t n)
{
	// here we need two matrices. Store it in one: 2N x N
	if (2*m!=n)
		SG_ERROR( "segment_loss should be 2 x quadratic matrix: %i!=%i\n", 2*m, n) ;

	m_num_segment_types = m;

	m_segment_loss.set_array(segment_loss, m, n/2, 2, true, true) ;
}

void CSegmentLoss::set_segment_ids(CArray<int32_t>* segment_ids)
{
	m_segment_ids = segment_ids;
}

void CSegmentLoss::set_segment_mask(CArray<float64_t>* segment_mask)
{
	m_segment_mask = segment_mask;
}

void CSegmentLoss::compute_loss(int32_t* all_pos, int32_t len)
{
#ifdef DEBUG
	SG_PRINT("compute loss: len: %i, m_num_segment_types: %i\n", len, m_num_segment_types);
	SG_PRINT("m_segment_mask->element(0):%f \n", m_segment_mask->element(0));
	SG_PRINT("m_segment_ids->element(0):%i \n", m_segment_ids->element(0));
#endif
	ASSERT(m_segment_ids->get_dim1()==len);
	ASSERT(m_segment_mask->get_dim1()==len);

	m_segment_loss_matrix.resize_array(m_num_segment_types,len);

	for (int seg_type=0; seg_type<m_num_segment_types; seg_type++)
	{
		float32_t value = 0;
		int32_t last_id = -1;
		int32_t last_pos = all_pos[len-1];
		for (int pos=len-1;pos>=0; pos--)
		{
			int32_t cur_id = m_segment_ids->element(pos);
			if (cur_id!=last_id)
			{
				// segment contribution
				value += m_segment_mask->element(pos)*m_segment_loss.element(cur_id, seg_type, 0);
				last_id = cur_id;
			}
			//length contribution
			value += m_segment_mask->element(pos)*m_segment_loss.element(cur_id, seg_type, 1)*(all_pos[pos]-last_pos);
			last_pos = all_pos[pos];
			m_segment_loss_matrix.element(seg_type, pos)=value;
		}
	}
#ifdef DEBUG 
	m_segment_loss_matrix.display_array();	
#endif
}
float32_t CSegmentLoss::get_segment_loss(int32_t from_pos, int32_t to_pos, int32_t segment_id)
{
	float32_t diff_contrib = m_segment_loss_matrix.element(segment_id, from_pos)-m_segment_loss_matrix.element(segment_id, to_pos);
	float32_t start_contrib = 0;

	//determine if the loss for the last segment 
	// has to be considered in this segment
	bool add_start_contrib=false;

	if (to_pos==m_segment_ids->get_array_size())
		add_start_contrib = true;
	else if (m_segment_ids->element(to_pos)!=m_segment_ids->element(to_pos+1))
		add_start_contrib = true;

	if (add_start_contrib)
	{
		start_contrib = m_segment_loss.element(segment_id, m_segment_ids->element(to_pos), 0)*m_segment_mask->element(to_pos);
	}	
#ifdef DEBUG
		SG_PRINT("segment_id:%i, from_pos:%i, to_pos:%i\n", segment_id, from_pos, to_pos);
		for (int i=from_pos; i<=to_pos; i++)
			SG_PRINT("%i ", m_segment_ids->element(to_pos));
		SG_PRINT(" \n");
		//SG_PRINT("m_segment_ids->element(from_pos):%i, m_segment_ids->element(to_pos):%i\n", m_segment_ids->element(from_pos), m_segment_ids->element(to_pos));
		SG_PRINT("diff_contrib:%f start_contrib:%f\n", diff_contrib, start_contrib);
#endif	
	return start_contrib + diff_contrib;
}
