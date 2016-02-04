

-- comment in lua
-- a = 10
-- print(a)

-- a = "akhil"
-- print(a)

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
mrDog = Dog:new()
-- mrDog:makeSound()

print(mrDog)
print(mrDog.sound)
