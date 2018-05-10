local LogSoftMax = torch.class('nn.LogSoftMax', 'nn.Module')

function LogSoftMax:updateOutput(input)
   input.THNN.LogSoftMax_updateOutput(
      input,
      self.output
   )
   return self.output
end

function LogSoftMax:updateGradInput(input, gradOutput)
   input.THNN.LogSoftMax_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      self.output
   )
   return self.gradInput
end
