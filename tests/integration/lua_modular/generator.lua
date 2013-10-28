package.path = package.path .. ";../../../src/interfaces/lua_modular/?.lua;"
package.cpath = package.cpath .. ";../../../src/interfaces/lua_modular/?.so;/usr/local/lib/?.so"

require("lfs")

example_dir = '../../examples/undocumented/lua_modular'
test_dir = '../../../testsuite/tests'
blacklist = {"load.lua", "MatrixTest.lua", "VectorTest.lua"}

function get_test_mod(tests)
	lfs.chdir(example_dir)
	local r = {}
	for _, v in pairs(tests) do
		local flag = 0
		if string.sub(v, -3) ~= "lua" then
			flag = 1
		end
		for _, n in pairs(blacklist) do
			if n == v then
				flag = 1
				break
			end
		end
		if flag == 0 then
			mod = string.sub(v, 1, string.len(v)-4)
			table.insert(r, mod)
	  end
	end
	return r
end
function setup_tests(tests)
	if not tests then
		local files={}
		for i in io.popen("ls " .. example_dir):lines() do
			table.insert(files, i)
		end
		table.sort(files)
		return files
	else
		return tests
	end
end

function generator(tests)
	r = get_test_mod(tests)
	for _, t in pairs(r) do
		require(t)
		for i = 1, #parameter_list do
			f = loadstring("a=" .. t .. "(unpack(parameter_list[" .. i .. "]))")
			f()
			print("OK")
		end
	end
end

tests = setup_tests(...)
generator(tests)
