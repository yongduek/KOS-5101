text = "ABCD가나다라"

for k, ch in enumerate(text):
    print(k, ch, ord(ch), hex(ord(ch)))

chtext = "王靖雯"

for c in chtext:
    print(c, ord(c), hex(ord(c)))
    
num = 0x4535
print(num, chr(num))