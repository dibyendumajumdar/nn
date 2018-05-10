local SpatialAdaptiveAveragePooling, parent = torch.class('nn.SpatialAdaptiveAveragePooling', 'nn.Module')

function SpatialAdaptiveAveragePooling:__init(W, H)
   parent.__init(self)

   self.W = W
   self.H = H
end

function SpatialAdaptiveAveragePooling:updateOutput(input)
   input.THNN.SpatialAdaptiveAveragePooling_updateOutput(
      input,
      self.output,
      self.W, self.H
   )
   return self.output
end

function SpatialAdaptiveAveragePooling:updateGradInput(input, gradOutput)
   input.THNN.SpatialAdaptiveAveragePooling_updateGradInput(
      input,
      gradOutput,
      self.gradInput
   )
   return self.gradInput
end

-- for backward compat
function SpatialAdaptiveAveragePooling:empty()
   self:clearState()
end

function SpatialAdaptiveAveragePooling:clearState()
   return parent.clearState(self)
end
