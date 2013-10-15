set $_exitcode = -1
run

if $_exitcode == -1
	thread apply all bt
	info locals
end

quit
