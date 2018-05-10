local Sqrt, parent = torch.class('nn.Sqrt','nn.Module')

function Sqrt:__init(b)
   parent.__init(self)
   self.eps = b or 0
end

function Sqrt:updateOutput(input)
   self.eps = self.eps or 0
   input.THNN.Sqrt_updateOutput(
      input,
      self.output,
      self.eps
   )
   return self.output
end

function Sqrt:updateGradInput(input, gradOutput)
   input.THNN.Sqrt_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      self.output
   )
   return self.gradInput
end
