#!/usr/bin/env python
import time
from modshogun import Time

parameter_list = [[5],[1.0]]
def library_time (sleep_secs):
	# measure wall clock time difference
	t=Time()
	time.sleep(sleep_secs)
	diff=t.cur_time_diff()

	# measure CPU time required
	cpu_diff=t.cur_runtime_diff_sec()

	# wall clock time should be above sleep_secs
	# but cpu time should be tiny
	#print diff, cpu_diff
	return diff>sleep_secs, cpu_diff<0.5

if __name__=='__main__':
	print('Time')
	library_time(*parameter_list[0])
