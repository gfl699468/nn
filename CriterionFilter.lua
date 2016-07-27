local CriterionFilter = torch.class('nn.CriterionFilter', 'nn.Criterion')

function CriterionFilter:__init(criterion, ignored_label)
  self.target = torch.zeros(1,2,2):long()
  self.criterion = {}
  self.ignored_label = torch.zeros(1):long()
  assert(ignored_label, 'No ignored label provided')
  self.ignored_label[1] = ignored_label
  self.ignored_label = self.ignored_label:long()
  assert(criterion, 'No criterion provided')
  self.criterion[1] = criterion

end

function CriterionFilter:updateOutput(input, target)
--TODO:The loss is wrong, maybe we can fix it with weight for some criterion.
  return self.criterion[1]:updateOutput(input, target)

end

function CriterionFilter:updateGradInput(input, target)

  self.gradInput = self.criterion[1]:updateGradInput(input, target)

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

function CriterionFilter:cuda()
  local tmp = self:type("torch.CudaTensor")
  tmp.ignored_label = tmp.ignored_label:cuda()
  return tmp
end
