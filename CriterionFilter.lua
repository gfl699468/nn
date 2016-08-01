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
  if type(target) == 'number' then
    self.target[1] = target
  elseif target:type() == 'torch.CudaTensor' then
    self.target = target
  else
    self.target = target:long()
  end --Save the target info so that it can be used in updateGradInput stage.
  if input:size(2) < self.ignored_label then
    if input:size(2) != (self.ignored_label - 1) then
      input.THNN.CriterionFilter_updateOutput(
        self.target:cdata(),
        input:cdata(),
        self.ignored_label:cdata(),
      )
    end
    local size = #input
    size[2] = size[2] + 1
    self.input = torch.Tensor(size):typeAs(input):fill(0)
    self.input:narrow(2,1,size[2]-1):copy(input)
    self.criterion[1].weights = torch.ones(size[2])
    self.criterion[1].weights[size[2]] = 0
    return self.criterion[1]:updateOutput(self.input, self.target)--Enlarge the input matrix and fill 0 to those additional space
  else
    return self.criterion[1]:updateOutput(input, self.target)
  end
end

function CriterionFilter:updateGradInput(input, target)
  self.gradInput = self.criterion[1]:updateGradInput(input, self.target)
  if input:size(2) < self.ignored_label then
    self.gradInput = self.gradInput:narrow(2,1,input:size(2))
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
