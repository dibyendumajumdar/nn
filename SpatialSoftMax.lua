local SpatialSoftMax, _ = torch.class('nn.SpatialSoftMax', 'nn.Module')

function SpatialSoftMax:updateOutput(input)
   input.THNN.SoftMax_updateOutput(
      input,
      self.output
   )
   return self.output
end

function SpatialSoftMax:updateGradInput(input, gradOutput)
   input.THNN.SoftMax_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      self.output
   )
   return self.gradInput
end
