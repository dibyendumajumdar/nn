local TemporalConvolution, parent = torch.class('nn.TemporalConvolution', 'nn.Module')

function TemporalConvolution:__init(inputFrameSize, outputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.outputFrameSize = outputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.bias = torch.Tensor(outputFrameSize)
   self.gradWeight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.gradBias = torch.Tensor(outputFrameSize)
   
   self:reset()
end

function TemporalConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.inputFrameSize)
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

function TemporalConvolution:updateOutput(input)
    input.THNN.TemporalConvolution_updateOutput(
	input, self.output,
	self.weight, self.bias,
	self.kW, self.dW,
	self.inputFrameSize, self.outputFrameSize
    )
   return self.output
end

function TemporalConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.THNN.TemporalConvolution_updateGradInput(
	  input, gradOutput,
	  self.gradInput, self.weight,
	  self.kW, self.dW
       )
      return self.gradInput
   end
end

function TemporalConvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input.THNN.TemporalConvolution_accGradParameters(
       input, gradOutput,
       self.gradWeight, self.gradBias,
       self.kW, self.dW, scale
   )
end

function TemporalConvolution:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end
