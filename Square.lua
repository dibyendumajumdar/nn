local Square, parent = torch.class('nn.Square', 'nn.Module')

function Square:__init(args)
   parent.__init(self)
end

function Square:updateOutput(input)
   input.THNN.Square_updateOutput(
      input,
      self.output
   )
   return self.output
end

function Square:updateGradInput(input, gradOutput)
   input.THNN.Square_updateGradInput(
      input,
      gradOutput,
      self.gradInput
   )
   return self.gradInput
end
