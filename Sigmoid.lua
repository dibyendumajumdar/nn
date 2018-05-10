local Sigmoid = torch.class('nn.Sigmoid', 'nn.Module')

function Sigmoid:updateOutput(input)
   input.THNN.Sigmoid_updateOutput(
      input,
      self.output
   )
   return self.output
end

function Sigmoid:updateGradInput(input, gradOutput)
   input.THNN.Sigmoid_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      self.output
   )
   return self.gradInput
end
