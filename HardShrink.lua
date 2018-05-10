local HardShrink, parent = torch.class('nn.HardShrink', 'nn.Module')

function HardShrink:__init(lam)
   parent.__init(self)
   self.lambda = lam or 0.5
end

function HardShrink:updateOutput(input)
   input.THNN.HardShrink_updateOutput(
      input,
      self.output,
      self.lambda
   )
   return self.output
end

function HardShrink:updateGradInput(input, gradOutput)
   input.THNN.HardShrink_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      self.lambda
   )
   return self.gradInput
end
