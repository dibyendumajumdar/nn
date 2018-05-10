local SoftMin, parent = torch.class('nn.SoftMin', 'nn.Module')

function SoftMin:updateOutput(input)
   self.mininput = self.mininput or input.new()
   self.mininput:resizeAs(input):copy(input):mul(-1)
   input.THNN.SoftMax_updateOutput(
      self.mininput,
      self.output
   )
   return self.output
end

function SoftMin:updateGradInput(input, gradOutput)
   self.mininput = self.mininput or input.new()
   self.mininput:resizeAs(input):copy(input):mul(-1)

   input.THNN.SoftMax_updateGradInput(
      self.mininput,
      gradOutput,
      self.gradInput,
      self.output
   )

   self.gradInput:mul(-1)
   return self.gradInput
end

function SoftMin:clearState()
   if self.mininput then self.mininput:set() end
   return parent.clearState(self)
end
