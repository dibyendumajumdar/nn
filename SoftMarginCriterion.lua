local SoftMarginCriterion, parent = torch.class('nn.SoftMarginCriterion', 'nn.Criterion')

function SoftMarginCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function SoftMarginCriterion:updateOutput(input, target)
   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.SoftMarginCriterion_updateOutput(
      input, target,
      self.output_tensor,
      self.sizeAverage)
   self.output = self.output_tensor[1]
   return self.output
end

function SoftMarginCriterion:updateGradInput(input, target)
   input.THNN.SoftMarginCriterion_updateGradInput(
      input, target,
      self.gradInput,
      self.sizeAverage)
   return self.gradInput
end
