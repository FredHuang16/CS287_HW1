

-- comment in lua
-- a = 10
-- print(a)

-- a = "akhil"
-- print(a)

function hingeLoss(z,y)
  local c, c_i, c_prime, c_prime_i

  -- get the value for the correct class
  c_i = torch.dot(y,torch.range(1,5))
  c = z[c_i]

  -- get the
  c_prime,c_prime_i = z:max(1)

  if c_prime_i[1] == c_i then
    z[c_i] = torch.min(z)
    c_prime,c_prime_i = z:max(1)
  end
  c_prime = c_prime[1]

  return math.max(0, 1 - (c - c_prime))
end

y = torch.Tensor({0,0,1,0,0})

z1 = torch.Tensor({1,2,3,4,5})
z2 = torch.Tensor({1,0,5,0,0})
z3 = torch.Tensor({4.5,0,5,0,0})

print(hingeLoss(z1,y))
print(hingeLoss(z2,y))
print(hingeLoss(z3,y))

function fact(n)
  if n == 1 then
    return 1
  else
    return n*fact(n-1)
  end
end

function print_table(t)
  for k,v in pairs(t) do
    print(k,v)
  end
end

Dog = {}

function Dog:makeSound()
  print("I say "..self.sound)
end

function Dog:new()
  newDog = {sound = "woof"}
  return setmetatable(newDog,{__index = self})
end

-- print_table(Dog)
-- mrDog = Dog:new()
-- mrDog:makeSound()

-- print(mrDog)
-- print(mrDog.sound)

-- implementation of f(x) = x^T A x + b^T x
function f(x,A,b)
  if buf == nil then
    buf = torch.Tensor() -- an empty tensor
  end
  buf:resize(A:size(1)) -- does a memory copy only if necesary
  ans = torch.dot(x,buf:mv(A,x))/2 + torch.dot(b,x)
  return ans
end

-- implementation of \nabla_x(f) = Ax + b
function dfdx(x,A,b)
  if grad == nil then
    grad = torch.Tensor()
  end
  grad:resizeAs(x)
  grad:mv(A,x):add(b)
  return grad
end

function onehot(len,location)
  local delta = torch.zeros(len)
  delta[location] = 1
  return delta
end



-- finite difference checking
function checkgrad(f,dfdx,x,A,b)
  -- first let's compute the gradient at our current point
  local grad = dfdx(x,A,b)
  -- now let's check it with finite differences
  local eps = 1e-5
  local xcopy = x:clone() -- might come in handy
  local len = grad:size(1)

  local epsVec = torch.zeros(len)
  for j = 1,len  do

    epsVec = epsVec:add(onehot(len,j)):mul(eps)
    -- print(epsVec)

    -- form finite difference: (f(x+eps,A,b) - f(x-eps,A,b))/(2*eps)
    finiteDiff = (f(x+epsVec,A,b) - f(x-epsVec,A,b))/(2*eps)

    -- now compare to our analytic gradient
    print(grad[j], finiteDiff)
    assert(torch.abs(grad[j]-finiteDiff) <= 1e-4)

    -- clean up
    epsVec:zero()
  end
end


function checkgrad(f,dfdx,x,param1,param2)

  -- first let's compute the gradient at our current point
  local grad = dfdx(x,param1,param2)
  local gradDims = grad:nDimension()

  if gradDims > 2 then
    print("More than 2 dimensions in the gradient. Feature not implemented")
    return -1
  end

  -- now let's check it with finite differences
  local eps = 1e-5
  local xcopy = x:clone() -- might come in handy

  local len1 = grad:size(1)
  local len2 = nil
  local epsVec = torch.zeros(x:size())

  if gradDims == 2 then
    len2 = grad:size(2)

    for j = 1,len1  do
      for k = 1,len2 do
        epsVec[j][k] = eps

        -- form finite difference: (f(x+eps,A,b) - f(x-eps,A,b))/(2*eps)
        f1 = crLoss(softmax(sparseMV(W+epsVec) + b),y)
        f2 = crLoss(softmax(sparseMV(W-epsVec) + b),y)
        finiteDiff = (f1 - f2)/(2*eps)
        -- finiteDiff = (f(x+epsVec,param1,param2) - f(x-epsVec,param1,param2))/(2*eps)

        -- now compare to our analytic gradient
        print(grad[j][k], finiteDiff)
        assert(torch.abs(grad[j][k]-finiteDiff) <= 1e-4)

        -- clean up
        epsVec:zero()
      end
    end
  else -- one dimentional x
    for j = 1,len1  do
      epsVec[j] = eps
      -- print(epsVec)

      -- form finite difference: (f(x+eps,A,b) - f(x-eps,A,b))/(2*eps)
      finiteDiff = (f(x+epsVec,param1,param2) - f(x-epsVec,param1,param2))/(2*eps)

      -- now compare to our analytic gradient
      print(grad[j], finiteDiff)
      assert(torch.abs(grad[j]-finiteDiff) <= 1e-4)

      -- clean up
      epsVec:zero()
    end
  end
end
