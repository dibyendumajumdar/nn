require 'THNNx'

local THNN = {}

THNN.NULL = nil

function THNN.getState()
   return nil
end

function THNN.optionalTensor(t)
   return t
end

THNN.kernels = {}
THNN.kernels['torch.FloatTensor'] = torch.getmetatable('torch.FloatTensor').THNN
THNN.kernels['torch.DoubleTensor'] = torch.getmetatable('torch.DoubleTensor').THNN

function THNN.runKernel(f, type, ...)
   local ftable = THNN.kernels[type]
   if not ftable then
      error('Unsupported tensor type: '..type)
   end
   local f = ftable[f]
   if not f then
      error(string.format("Function '%s' not found for tensor type '%s'.", f, type))
   end
   f(...)
end

return THNN
