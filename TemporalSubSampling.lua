local TemporalSubSampling, parent = torch.class('nn.TemporalSubSampling', 'nn.Module')

function TemporalSubSampling:__init(inputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(inputFrameSize)
   self.bias = torch.Tensor(inputFrameSize)
   self.gradWeight = torch.Tensor(inputFrameSize)
   self.gradBias = torch.Tensor(inputFrameSize)
   
   self:reset()
end

function TemporalSubSampling:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function TemporalSubSampling:updateOutput(input)
    input.THNN.TemporalSubSampling_updateOutput(
	input, self.output,
	self.weight, self.bias,
	self.kW, self.dW, self.inputFrameSize
    )
   return self.output
end

function TemporalSubSampling:updateGradInput(input, gradOutput)
    if self.gradInput then
	input.THNN.TemporalSubSampling_updateGradInput(
	    input, gradOutput, self.gradInput,
	    self.weight, self.kW, self.dW
	)
	return self.gradInput
   end
end

function TemporalSubSampling:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    input.THNN.TemporalSubSampling_accGradParameters(
	input, gradOutput, self.gradWeight,
	self.gradBias, self.kW, self.dW, scale
    )
end
