local SoftMax, _ = torch.class('nn.SoftMax', 'nn.Module')

function SoftMax:updateOutput(input)
   input.THNN.SoftMax_updateOutput(
      input,
      self.output
   )
   return self.output
end

function SoftMax:updateGradInput(input, gradOutput)
   input.THNN.SoftMax_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      self.output
   )
   return self.gradInput
end
