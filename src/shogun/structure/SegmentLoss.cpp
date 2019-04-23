#include <stdio.h>
#include <string.h>

#include <shogun/mathematics/Math.h>
#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/structure/SegmentLoss.h>
#include <shogun/base/SGObject.h>
//# define DEBUG

using namespace shogun;

SegmentLoss::SegmentLoss()
	:SGObject(),
	m_segment_loss_matrix(1,1),
	m_segment_loss(1,1,2),
	m_num_segment_types(0)
{
}
SegmentLoss::~SegmentLoss()
{
}

void SegmentLoss::set_segment_loss(float64_t* segment_loss, int32_t m, int32_t n)
{
	// here we need two matrices. Store it in one: 2N x N
	if (2*m!=n)
		error("segment_loss should be 2 x quadratic matrix: {}!={}", 2*m, n);

	m_num_segment_types = m;

	m_segment_loss.set_array(segment_loss, m, n/2, 2, true, true) ;
}

void SegmentLoss::set_segment_ids(const std::vector<int32_t>& segment_ids)
{
	m_segment_ids = segment_ids;
}

void SegmentLoss::set_segment_mask(const std::vector<float64_t>& segment_mask)
{
	m_segment_mask = segment_mask;
}

void SegmentLoss::compute_loss(int32_t* all_pos, int32_t len)
{
#ifdef DEBUG
	io::print("compute loss: len: {}, m_num_segment_types: {}\n", len, m_num_segment_types);
	io::print("m_segment_mask->element(0):{} \n", m_segment_mask.at(0));
	io::print("m_segment_ids->element(0):{} \n", m_segment_ids.at(0));
#endif
	ASSERT(m_segment_ids.size()==len)
	ASSERT(m_segment_mask.size()==len)

	m_segment_loss_matrix.resize_array(m_num_segment_types,len);

	for (int seg_type=0; seg_type<m_num_segment_types; seg_type++)
	{
		float32_t value = 0;
		int32_t last_id = -1;
		int32_t last_pos = all_pos[len-1];
		for (int pos=len-1;pos>=0; pos--)
		{
			int32_t cur_id = m_segment_ids.at(pos);
			if (cur_id!=last_id)
			{
				// segment contribution
				value += m_segment_mask.at(pos)*m_segment_loss.element(cur_id, seg_type, 0);
				last_id = cur_id;
			}
			//length contribution (nucleotide loss)
			value += m_segment_mask.at(pos)*m_segment_loss.element(cur_id, seg_type, 1)*(last_pos-all_pos[pos]);
			last_pos = all_pos[pos];
			m_segment_loss_matrix.element(seg_type, pos)=value;
		}
	}
#ifdef DEBUG
	m_segment_loss_matrix.display_array();
#endif
}

