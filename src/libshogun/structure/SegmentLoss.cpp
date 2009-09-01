#include <stdio.h>
#include <string.h>

#include "lib/Mathematics.h"
#include "lib/config.h"
#include "lib/io.h"
#include "structure/SegmentLoss.h"

CSegmentLoss::CSegmentLoss()
	:CSGObject(),
	m_segment_ids(1),
	m_segment_mask(1),
	m_segment_loss(1,1,2),
	m_num_segment_types(NULL),
	m_segment_loss_matrix(1,1)
{}


void CSegmentLoss::set_segment_loss(
	float64_t* segment_loss, int32_t m, int32_t n)
{
	// here we need two matrices. Store it in one: 2N x N
	if (2*m!=n)
		SG_ERROR( "segment_loss should be 2 x quadratic matrix: %i!=%i\n", 2*m, n) ;

	if (m!=m_max_a_id+1)
		SG_ERROR( "segment_loss size should match m_max_a_id: %i!=%i\n", m, m_max_a_id+1) ;

	m_segment_loss.set_array(segment_loss, m, n/2, 2, true, true) ;
	/*for (int32_t i=0; i<n; i++)
		for (int32_t j=0; j<n; j++)
		SG_DEBUG( "loss(%i,%i)=%f\n", i,j, m_segment_loss.element(0,i,j)) ;*/
}

void CSegmentLoss::set_segment_ids(CArray<int32_t> *segment_ids)
{
	SG_UNREF(m_segment_ids);
	m_segment_ids = segment_ids;
	SG_REF(m_segment_ids);
}

void CSegmentLoss::set_segment_mask(CArray<float64_t> *segment_mask)
{
	SG_UNREF(m_segment_mask);
	m_segment_mask = segment_mask;
	SG_REF(m_segment_mask);
}

void CSegmentLoss::compute_loss(int32_t* all_pos, int32_t* len)
{
	ASSERT(m_segment_ids.get_dim1()==len);
	ASSERT(m_segment_mask.get_dim1()==len);

	m_segment_loss_matrix.resize_arary(m_num_segment_types,len);

	for (int seg_type=0; seg_type<m_num_segment_types; seg_type++)
	{
		float32_t value = 0;
		int32_t last_id = m_segment_ids.get_element(0);
		int32_t last_pos = all_pos[0];
		for (int pos=0; pos<len; pos++)
		{
			int32_t cur_id = m_segment_ids.element(pos);
			if (cur_id!=last_id)
			{
				// segment contribution
				value += m_segment_mask.element(pos)*m_segment_loss.element(cur_id, seg_type, 0);
				last_id = cur_id;
			}
			//length contribution
			value += m_segment_mask.element(pos)*m_segment_loss.element(cur_id, seg_type, 1)*(all_pos[pos]-last_pos);
			last_pos = all_pos[pos];
			m_segment_loss_matrix.element(seg_type, pos)=value;
		}
	}
}
float32_t CSegmentLoss::get_segment_loss(int32_t from_pos, int32_t to_pos, int32_t segment_id)
{
	return m_segment_loss_matrix(segment_id, to_pos)-m_segment_loss_matrix(segment_id, from_pos);
}
