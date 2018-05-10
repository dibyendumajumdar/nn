local Tanh = torch.class('nn.Tanh', 'nn.Module')

function Tanh:updateOutput(input)
   input.THNN.Tanh_updateOutput(
      input,
      self.output
   )
   return self.output
end

function Tanh:updateGradInput(input, gradOutput)
   input.THNN.Tanh_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      self.output
   )
   return self.gradInput
end
