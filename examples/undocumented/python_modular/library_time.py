import time
from modshogun import Time


# measure wall clock time difference
t=Time()
time.sleep(1)
diff=t.cur_time_diff()

# measure CPU time required

cpu_diff=t.cur_runtime_diff_sec()
