print("Hellow world")

taxi = True

if taxi == True:
    print("take taxi you bougee")
else:
    print("walk you poor fuck")

text_list = ['one', 'two', 'three']

marks = [90, 25, 67, 45, 80]
number = 0
for mark in marks:
    number = number + 1 
    if mark >= 60: 
        print("%d번 학생은 합격입니다." % number)
    else: 
        print("%d번 학생은 불합격입니다." % number)


def add(a, b, c):
    abc = a + b + c
    return abc

a = 1
b = 2
c = 3

total = add(a,b,c)
print("total: ", total)

def aa(a):
    input("enter man:")

i = input("enter any number:")
print(i, type(i))
num_i = int(i)

total = add(a,b,num_i)
print(total)

filename = "./26-1/gks.txt"
with open(filename, "r", encoding="utf-8") as fd:
    tt = fd.readlines()

print(tt)

print(type(tt), len(tt), type(tt[0]))

for i,line in enumerate(tt):
    line = line.strip()
    print(f"{i}: {len(line)} {line}")

l5 = tt[6].strip()
for i, c in enumerate(l5):
    print(f"{i}:{ord(c)} {c}")