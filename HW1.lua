-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
-- ...


function main()
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')

   -- load the data
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]

   train_input = f:read('train_input'):all()
   train_output = f:read('train_output'):all()

   valid_input = f:read('valid_input'):all()
   valid_output = f:read('valid_output'):all()

   test_input = f:read('test_input'):all()

   W = torch.DoubleTensor(nfeatures,nclasses)
   b = torch.DoubleTensor(nclasses)

   local alpha = torch.linspace(0.01,2.5,50)
   local accuracy = torch.Tensor(50):zero()

   for i=1,alpha:size(1) do

     -- Train.
     -- print("Alpha = ",alpha)
     nbTrain(alpha[i])

     -- Validate
     _ , accuracy[i] = nbTest("valid")
   end

   _ , best_alpha_idx = torch.max(accuracy)

   -- run on test data
   nbTrain(alpha[best_alpha_idx])
   y_hat, _ = nbTest("test")

   -- write to file
   write_to_file({["alpha"] = alpha,["accuracy"] = accuracy
      ,["output"] = y_hat},"output."..opt.datafile)
end

function getFullVec(x_sparse)
  local x_full = torch.Tensor(nfeatures):zero()

  for i=1,nfeatures do

    -- if we've reached the padding then exit
    if x_sparse[i] == 1 then break end

    -- else set the feature to 1
    x_full[x_sparse[i] - 1] = 1
  end

  return x_full
end

function getRawCounts()
  -- set up initial variables
  local num_samples = train_input:size(1)
  local max_sent_len = train_input:size(2)

  -- not local
  W_count = torch.DoubleTensor(nfeatures,nclasses):zero()
  b_count = torch.DoubleTensor(nclasses):zero()


  -- count the number of each output class
  for i=1,num_samples do
    b_count[train_output[i]] = b_count[train_output[i]] + 1
  end

  -- print(b,num_samples,max_sent_len)

  -- get the counts for each token
  for i=1,num_samples do
    for j=1,max_sent_len do

      -- if we've reached the padding, go to the next sample
      if train_input[i][j] == 1 then
        break
      end

      -- get the index of the current word
      -- subtract one since 1 is padding
      feature_idx = train_input[i][j] - 1
      output_class = train_output[i]

      -- update the count matrix
      W_count[feature_idx][output_class] = W_count[feature_idx][output_class] + 1
    end
  end
  return 0
end

function nbTrain(alpha)

  -- default for alpha
  alpha = alpha or .001

  -- if we haven't generated the counts then do so
  if not W_count then getRawCounts() end

  -- add alpha to all the counts
  -- W_master = W:clone()
  torch.add(W,W_count,alpha)

  -- normalize W to be probabilities
  for i = 1,nclasses do
    W:select(2,i):div(b_count[i] + alpha*nfeatures)
  end
  W:log()

  -- normalize b and take log
  torch.div(b,b_count,train_input:size(1)):log()
end

function nbTest(type)
  local num_samples_test
  local max_sent_len_test
  local z
  local y_hat
  local X
  local y

  if type == "valid" then
    num_samples_test = valid_input:size(1)
    max_sent_len_test = valid_input:size(2)
    X = valid_input
    y = valid_output
  elseif type == "test" then
    num_samples_test = test_input:size(1)
    max_sent_len_test = test_input:size(2)
    X = test_input
  end

  z = torch.Tensor(num_samples_test,nclasses)
  y_hat = torch.Tensor(num_samples_test)

  for i=1,num_samples_test do
    z:select(1,i):mv(W:t(),getFullVec(X:select(1,i))):add(b)
    _ , y_hat[i] = torch.max(z:select(1,i),1)
  end

  local correct = 0
  if type == "valid" then
    -- print(torch.typename(y_hat_int),torch.typename(y))
    -- print(y_hat)
    correct = torch.sum(torch.eq(y_hat:int(), y))
    print("Num correct = ",correct / num_samples_test)
  end

  return y_hat , correct / num_samples_test
end

function write_to_file(obj,f)
  local myFile = hdf5.open(f, 'w')
  for k,v in pairs(obj) do myFile:write(k, v) end
  myFile:close()
end

main()
