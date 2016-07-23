local CriterionFilter = torch.class('nn.CriterionFilter')

function CriterionFilter:__init(criterion, ignored_label)
  self.target = torch.zeros(1):long()
  self.criterion = torch.zeros(1):long()
  assert(ignored_label, 'No ignored label provided')
  self.ignored_label = ignored_label
  assert(criterion, 'No criterion provided')
  self.criterion[1] = criterion

end

function CriterionFilter:updateOutput(input, target)

  return self.criterion:updateOutput(input, target)

end

function CriterionFilter:updateGradInput(input, target)

  self.gradInput = self.criterion:updateGradInput(input, target)

  if type(target) == 'number' then
     self.target[1] = target
  elseif target:type() == 'torch.CudaTensor' then
     self.target = target
  else
     self.target = target:long()
  end

  input.THNN.CriterionFilter_updateGradInput(
     self.target:cdata(),
     self.gradInput:cdata(),
     self.ignored_label:cdata()
  )

  return self.gradInput

end
