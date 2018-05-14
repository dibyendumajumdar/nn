-- Simple utility to convert from
-- C prototypes in THNN_H.lua to 
-- cwrap interface. 
-- Makes many assumptions about how the
-- prototypes are presented.
-- Cannot handle 'THTensor*' versus 'THTensor *', pattern matching needs work
-- Processes // [OPTIONAL] markers
-- To run: ravi extract.lua < ../../../THNN_h.lua > outfile

local mapping = {
	['THTensor *'] = 'Tensor',
	['int '] = '"int"',
	['THIndexTensor *'] = '"IndexTensor"',
	['THIntegerTensor *'] = '"IntTensor"',	
	['bool '] = '"boolean"',
	['accreal '] = 'accreal',
	['int64_t '] = '"int64_t"',
	['THIndex_t '] = '"index"',
	['THGenerator *'] = '"Generator"',
	['double '] = '"double"',
}

while true do
	local line = io.read()
	if line == nil then break end

	if string.match(line, "^TH_API") then
		local name = string.match(line, "^TH_API void THNN_%(([%w_]+)%)%(")
		print('     wrap("' .. name .. '",')
        print('        cname("' .. name .. '"),')
        print('        {{name=THNNState, invisible=true, default="NULL"},')
		line = io.read()
        while true do
			line = io.read()			
			if line == nil then break end
			local optional = false
			if string.match(line, '%[OPTIONAL%]') then
				optional = true
			end
			line = string.gsub(line, '//.*', '')
			for typename, varname in string.gmatch(line, "([%w_]+[ ][%*]?)[ ]-([%w]+)[,%)]?") do
				local t = mapping[typename]
				assert(t, 'typename [' .. typename .. '] not found')
				if t == '"IndexTensor"' then
					if optional then
						io.write('        {name=' .. t .. ', noreadadd=true, optional=true},')
					else
						io.write('        {name=' .. t .. ', noreadadd=true},')
					end
				elseif optional then
					io.write('        {name=' .. t .. ', optional=true},')
				else
					io.write('        {name=' .. t .. '},')
				end
			end
			if string.match(line, ".*%);.*") then
				io.write('})\n')
				break
			else
				io.write('\n')
			end
		end
	end
end

