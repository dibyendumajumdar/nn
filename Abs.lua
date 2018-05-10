local Abs, parent = torch.class('nn.Abs', 'nn.Module')

function Abs:__init()
   parent.__init(self)
end

function Abs:updateOutput(input)
   input.THNN.Abs_updateOutput(
      input,
      self.output
   )
   return self.output
end

function Abs:updateGradInput(input, gradOutput)
   input.THNN.Abs_updateGradInput(
      input,
      gradOutput,
      self.gradInput
   )
   return self.gradInput
end
