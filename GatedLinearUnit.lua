local GatedLinearUnit, parent = torch.class('nn.GatedLinearUnit', 'nn.Module')

function GatedLinearUnit:__init(dim)
   parent.__init(self)
   self.dim = dim
end

function GatedLinearUnit:updateOutput(input)
   local dim = self.dim or input:dim()
   input.THNN.GatedLinear_updateOutput(
      input,
      self.output,
      dim
   )
   return self.output
end

function GatedLinearUnit:updateGradInput(input, gradOutput)
   local dim = self.dim or input:dim()
   input.THNN.GatedLinear_updateGradInput(
      input,
      gradOutput,
      self.gradInput,
      dim
   )
   return self.gradInput
end
