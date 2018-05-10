local MarginCriterion, parent = torch.class('nn.MarginCriterion', 'nn.Criterion')

function MarginCriterion:__init(margin)
   parent.__init(self)
   self.sizeAverage = true
   self.margin = margin or 1
end

function MarginCriterion:updateOutput(input, target)
   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.MarginCriterion_updateOutput(
      input,
      target,
      self.output_tensor,
      self.sizeAverage,
      self.margin
   )
   self.output = self.output_tensor[1]
   return self.output
end

function MarginCriterion:updateGradInput(input, target)
   input.THNN.MarginCriterion_updateGradInput(
      input,
      target,
      self.gradInput,
      self.sizeAverage,
      self.margin
   )
   return self.gradInput
end
