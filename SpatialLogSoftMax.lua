local SpatialLogSoftMax = torch.class('nn.SpatialLogSoftMax', 'nn.Module')

function SpatialLogSoftMax:updateOutput(input)
   input.THNN.LogSoftMax_updateOutput(
      input,
      self.output
   )
   return self.output
end

function SpatialLogSoftMax:updateGradInput(input, gradOutput)
   input.THNN.LogSoftMax_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      self.output
   )
   return self.gradInput
end
