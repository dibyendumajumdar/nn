local THNN = require 'nn.THNN'
local L1Cost, parent = torch.class('nn.L1Cost','nn.Criterion')

function L1Cost:__init()
   parent.__init(self)
end

function L1Cost:updateOutput(input)
   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.L1Cost_updateOutput(
      input,
      self.output_tensor
   )
   self.output = self.output_tensor[1]
   return self.output
end

function L1Cost:updateGradInput(input)
   input.THNN.L1Cost_updateGradInput(
      input,
      THNN.NULL,
      self.gradInput
   )
   return self.gradInput
end

function L1Cost:clearState()
   if self.output_tensor then self.output_tensor:set() end
   return parent.clearState(self)
end
