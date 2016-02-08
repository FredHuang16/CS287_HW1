-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
cmd:option('-alpha',1,'alpha hyperparameter for nb')
cmd:option('-lambda',0.5,'lambda hyperparameter for lg')
cmd:option('-M',100,'mini-batch size hyperparameter for lg')
cmd:option('-eta',0.5,'learning rate hyperparameter for lg')
cmd:option('-N',100,'num epochs hyperparameter for lg')

-- ...


function main()
   -- Parse input params
   opt = cmd:parse(arg)
   if opt.classifier == "test" then testCode(); return end

   local f = hdf5.open(opt.datafile, 'r')

   -- load the data
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]

   train_input = f:read('train_input'):all()
   train_output = f:read('train_output'):all()

   valid_input = f:read('valid_input'):all()
   valid_output = f:read('valid_output'):all()

   test_input = f:read('test_input'):all()

   W = torch.DoubleTensor(nclasses,nfeatures)
   b = torch.DoubleTensor(nclasses)

   if opt.classifier == "nb" then nbRun()
   elseif opt.classifier == "lg" then lgRun()
   elseif opt.classifier == "svm" then svmRun()
   else print("Classifier should be (nb,lg,svm)")
   end
end

-- Logistic Regression Fucntions
function lgRun()
  print("Running Multiclass Logistic Regression on ",opt.datafile)
  lgTrain()
  test("valid")
  y_hat, _ = test("test")
  writeToFile({["output"] = y_hat},"output."..opt.datafile)
end

function lgTrain()
  sgd(crLoss,dLcrdW,dLcrdb)
end

function l_softmax(vec)
  local M = torch.max(vec)
  local vec2 = torch.add(vec,-M):exp()
  local denom = math.log(vec2:sum()) + M
  return vec - denom
end

function crLoss(x,y)
  local z = sparseMV(W,x) + b
  local l_y_hat = l_softmax(z)

  for i=1,nclasses do
    if y[i] == 1 then return -l_y_hat[i] end
  end
  return "error"
end

function dLcrdz(x,y)
  local z = sparseMV(W,x) + b
  local l_y_hat = l_softmax(z)
  local dz = torch.Tensor(nclasses)
  for i=1,nclasses do
    if y[i] == 1 then dz[i] = -(1-math.exp(l_y_hat[i]))
    else dz[i] = math.exp(l_y_hat[i])
    end
  end
  return dz
end

function dLcrdb(x,y)
  return dLcrdz(x,y)
end

function dLcrdW(x,y)
  local dz = dLcrdz(x,y)
  local dW = torch.Tensor(nclasses,nfeatures):zero()

  for j=1,x:size(1) do
    if(x[j] == 1) then break end -- we've reached the padding so quit
    dW[{{},x[j]-1}] = dz -- else set the derivative to be dLcrdz
  end
  return dW
end

-- SVM functions

function svmRun()
  print("Running Linear SVM on ",opt.datafile)
  print("Running SVM on ",opt.datafile)
  svmTrain()
  test("valid")
  -- y_hat, _ = test("test")
  -- writeToFile({["output"] = y_hat},"output."..opt.datafile)
end

function svmTrain()
  sgd(hingeLoss,dLhdW,dLhdb)
end

function dLhdz(x,y)
  local dz = torch.zeros(nclasses)
  local z = sparseMV(W,x) + b
  local c, c_i, c_prime,c_prime_i

  -- get the value for the correct class
  c_i = torch.dot(y,torch.range(1,nclasses))
  c = z[c_i]
  c_prime,c_prime_i = getCPrime(z,c_i)

  if c - c_prime < 1 then
    dz[c_i] = -1
    dz[c_prime_i] = 1
  end

  return dz
end

function dLhdW(x,y)
  local dz = dLhdz(x,y)
  local dW = torch.Tensor(nclasses,nfeatures):zero()

  for j=1,x:size(1) do
    if(x[j] == 1) then break end -- we've reached the padding so quit
    dW[{{},x[j]-1}] = dz -- else set the derivative to be dLcrdz
  end
  return dW
end

function dLhdb(x,y)
  return dLhdz(x,y)
end

function hingeLoss(x,y)
  local z = sparseMV(W,x) + b
  local c, c_i, c_prime

  -- get the value for the correct class
  c_i = torch.dot(y,torch.range(1,nclasses))
  c = z[c_i]
  c_prime,_ = getCPrime(z,c_i)

  return math.max(0, 1 - (c - c_prime))
end

function getCPrime(z,c_i)
  local c_prime, c_prime_i
  c_prime,c_prime_i = z:max(1)

  if c_prime_i[1] == c_i then
    z[c_i] = torch.min(z)
    c_prime,c_prime_i = z:max(1)
  end
  return c_prime[1],c_prime_i[1]
end

-- Naive Bayes functions
function nbRun()
  print("Running Naive Bayes on ",opt.datafile)
  -- check if alpha was specified if so use it

  if(opt.alpha ~= -1) then
    nbTrain(opt.alpha)
    y_hat, _ = test("test")
    writeToFile({["output"] = y_hat},"output."..opt.datafile)

  -- if not then loop over some reasonable range of alphas
  else
    print("Alpha not specified. Grid search over alpha")
    local alphaVec = torch.linspace(0.0001,5,25)
    local accuracy = torch.Tensor(25):zero()

    for i=1,alphaVec:size(1) do

      -- Train.
      -- print("Alpha = ",alpha)
      nbTrain(alphaVec[i])

      -- Validate
      _ , accuracy[i] = test("valid")
    end

    _ , best_alpha_idx = torch.max(accuracy)

    -- run on test data
    nbTrain(alphaVec[best_alpha_idx])
    y_hat, _ = test("test")

    -- write to file
    writeToFile({["alpha"] = alphaVec,["accuracy"] = accuracy
       ,["output"] = y_hat},"output."..opt.datafile)
  end
end

function getNBParams()
  -- set up initial variables
  local num_samples = train_input:size(1)
  local max_sent_len = train_input:size(2)

  -- not local
  W_count = torch.DoubleTensor(nclasses,nfeatures):zero()

  -- local
  local b_count = torch.DoubleTensor(nclasses):zero()

  -- count the number of each output class
  for i=1,num_samples do
    b_count[train_output[i]] = b_count[train_output[i]] + 1
  end

  -- normalize b and take log
  torch.div(b,b_count,train_input:size(1)):log()
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
      W_count[output_class][feature_idx] = W_count[output_class][feature_idx] + 1
    end
  end

  -- divide by the number of counts in each row
  W_rowsums = W_count:sum(2):squeeze()

  return 0
end

function nbTrain(alpha)

  -- default for alpha
  alpha = alpha or .001

  -- if we haven't generated the counts then do so
  if not W_count then getNBParams() end

  -- add alpha to all the counts
  torch.add(W,W_count,alpha)

  -- normalize W to be probabilities
  for i = 1,nclasses do
    W:select(1,i):div(W_rowsums[i] + nfeatures * alpha)
  end

  -- print(W:sum(2))
  W:log()
end

-- funciton used to run the trained parameters on the validation set
-- and the test set
function test(type)
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

  z = torch.Tensor(num_samples_test,nclasses):zero()
  y_hat = torch.Tensor(num_samples_test)

  -- loop over samples
  for i=1,num_samples_test do
    -- loop over classes
    for j=1,nclasses do
      z[i][j] = b[j] -- add in b

      -- loop over features
      for k=1,max_sent_len_test do
        if X[i][k] == 1 then break end -- we've reached the padding

        z[i][j] = z[i][j] + W[j][X[i][k]-1] -- 1 is padding
      end
    end
    _ , y_hat[i] = torch.max(z:select(1,i),1)
  end

  local correct = 0
  if type == "valid" then
    -- print(torch.typename(y_hat_int),torch.typename(y))
    -- print(y_hat)
    correct = torch.sum(torch.eq(y_hat:int(), y))
    print("Accuracy = ",correct / num_samples_test)
  end

  return y_hat , correct / num_samples_test
end

-- utilities
function sgd(loss,grad_W,grad_b)
  local num_samples = train_input:size(1)
  local max_sent_len = train_input:size(2)

  W:zero()
  b:zero()
  local dW = torch.Tensor(nclasses,nfeatures)
  local db = torch.Tensor(nclasses)

  local x = torch.Tensor()
  local z = torch.Tensor()
  local l_y_hat = torch.Tensor()
  local y = torch.Tensor()
  local L = torch.zeros(opt.N)
  local idx = 0

  local adjFactorW = 1 - (opt.eta * opt.lambda)/(nclasses * nfeatures)
  local adjFactorb = 1 - (opt.eta * opt.lambda)/nclasses
  print("adjFactorW",adjFactorW,"adjFactorb",adjFactorb)

  for i=1,opt.N do -- number of epochs
    dW:zero()
    db:zero()

    for j=1,opt.M do -- number of samples used to compute the gradient
      idx = torch.random(num_samples)
      x = train_input[idx]
      y = oneHot(train_output[idx],nclasses)
      L[i] = L[i] + loss(x,y)

      --checkgrad_W(dW_update,y,x)
      dW:add(grad_W(x,y):div(1/opt.M))
      db:add(grad_b(x,y):div(1/opt.M))
    end

    W:mul(adjFactorW):add(-dW:mul(opt.eta))
    b:mul(adjFactorb):add(-db:mul(opt.eta))

    print("i = ",i,"Loss = ",L[i])
  end

  return L
end

function checkgrad_W(dW,y,x)

  -- now let's check it with finite differences
  local eps = 1e-5
  local len1 = dW:size(1)
  local len2 = dW:size(2)
  local epsVec = torch.zeros(W:size())

  for j = 1,len1  do
    for k = 1,len2 do
      epsVec[j][k] = eps

      -- form finite difference: (f(x+eps,A,b) - f(x-eps,A,b))/(2*eps)
      z1 = sparseMV(W+epsVec,x) + b
      y_hat1 = softmax(z1)

      z2 = sparseMV(W-epsVec,x) + b
      y_hat2 = softmax(z2)
      f1 = crLoss(y_hat1,y)
      f2 = crLoss(y_hat2,y)

      finiteDiff = (f1 - f2)/(2*eps)
      -- finiteDiff = (f(x+epsVec,param1,param2) - f(x-epsVec,param1,param2))/(2*eps)

      -- now compare to our analytic gradient
      if torch.abs(dW[j][k]-finiteDiff) > 1e-4 then
        print("j",j,"k",k)
        print("z1",z1)
        print("y_hat1",y_hat1)
        print("z2",z2)
        print("y_hat2",y_hat2)
        print("y",y)
        print(dW[j][k], finiteDiff)
        assert(torch.abs(dW[j][k]-finiteDiff) <= 1e-4)
      end

      -- clean up
      epsVec:zero()
    end
  end
end

function oneHot(h,len)
  local h_vec = torch.Tensor(len):zero()
  h_vec[h] = 1
  return h_vec
end

function sparseMV(Mat,vec)
  local s1 = Mat:size(1)
  local z = torch.Tensor(s1):zero()
  for j=1,s1 do
    for k=1,vec:size(1) do
      if vec[k] == 1 then break end
      z[j] = z[j] + Mat[j][vec[k]-1]
    end
  end
  return z
end

function writeToFile(obj,f)
  local myFile = hdf5.open(f, 'w')
  for k,v in pairs(obj) do myFile:write(k, v) end
  myFile:close()
end

-- testing beer
function testCode()
  -- let's define some global memory we'll update, and some fixed, global parameters
  buf = nil
  grad = nil

  torch.manualSeed(287)
  D = 3 -- dimensionality of x
  A = torch.randn(D,D)
  -- ensure symmetric (note this does a memory copy!)
  A = A + A:t()
  b = torch.randn(D)
  x = torch.randn(D)

  print(A,x,dfdA(A,x,b))
  print(checkgrad(f,dfdA,A,x,b))
end
main()
