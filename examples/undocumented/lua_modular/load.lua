function load_numbers(fname)
	f = io.open(fname)
	if (f == nil) then
		print("Cannot open ", fname)
	end
	matrix = {}
	while true do
		local	 num = f:read("*number")
		if not num then break end
		table.insert(matrix, num)
	end
	f:close()
	n = #matrix
	result = {}
	for i = 1, 2 do
		result[i] = {}
		for j = 1, n/2 do
			result[i][j] = matrix[(i-1) * (n/2) + j]
		end
	end
	return result
end

function load_labels(fname)
	f = io.open(fname)
	if (f == nil) then
		print("Cannot open ", fname)
	end
	matrix = {}
	while true do
		local	 num = f:read("*number")
		if not num then break end
		table.insert(matrix, num)
	end
	f:close()
	return matrix
end

function load_dna(fname)
	f = io.open(fname)
	if (f == nil) then
		print("Cannot open file:", fname)
	end
	matrix = {}
	while true do
		print("love")
		local	 num = f:read("*line")
		if not num then break end
		table.insert(matrix, num)
	end
	f:close()
	for k, v in pairs(matrix) do print (v) end
	return matrix
end

function load_cubes(fname)
	f = io.open(fname)
	if (f == nil) then
		print("Cannot open file:", fname)
	end
	matrix = {}
	while true do
		local	 num = f:read("*line")
		if not num then break end
		table.insert(matrix, num)
	end
	f:close()
	for k, v in pairs(matrix) do print (v) end
	return matrix
end
