local LogSigmoid, parent = torch.class('nn.LogSigmoid', 'nn.Module')

function LogSigmoid:updateOutput(input)
   self.buffer = self.buffer or input.new()
   input.THNN.LogSigmoid_updateOutput(
      input,
      self.output,
      self.buffer
   )
   return self.output
end

function LogSigmoid:updateGradInput(input, gradOutput)
   input.THNN.LogSigmoid_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      self.buffer
   )
   return self.gradInput
end

function LogSigmoid:clearState()
   if self.buffer then self.buffer:set() end
   return parent.clearState(self)
end

