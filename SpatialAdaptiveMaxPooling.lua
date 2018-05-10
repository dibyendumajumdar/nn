local SpatialAdaptiveMaxPooling, parent = torch.class('nn.SpatialAdaptiveMaxPooling', 'nn.Module')

function SpatialAdaptiveMaxPooling:__init(W, H)
   parent.__init(self)

   self.W = W
   self.H = H
end

function SpatialAdaptiveMaxPooling:updateOutput(input)
   self.indices = self.indices or torch.LongTensor()
   if torch.typename(input):find('torch%.Cuda.*Tensor') then
      self.indices = torch.CudaLongTensor and self.indices:cudaLong() or self.indices
   else
      self.indices = self.indices:long()
   end
   input.THNN.SpatialAdaptiveMaxPooling_updateOutput(
      input,
      self.output,
      self.indices,
      self.W, self.H
   )
   return self.output
end

function SpatialAdaptiveMaxPooling:updateGradInput(input, gradOutput)
   input.THNN.SpatialAdaptiveMaxPooling_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      self.indices
   )
   return self.gradInput
end

-- for backward compat
function SpatialAdaptiveMaxPooling:empty()
   self:clearState()
end

function SpatialAdaptiveMaxPooling:clearState()
   if self.indices then
      self.indices:set()
   end
   return parent.clearState(self)
end
