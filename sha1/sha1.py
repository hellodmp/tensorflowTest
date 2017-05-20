
import hashlib


if __name__ == "__main__":
    fp = open("./sha1.txt","w")
    for i in range(1,1000000):
        key = str(i)
        value = ""
        for c in hashlib.sha1(key).digest():
            value += str(ord(c))+","
        tempt = i
        while tempt/10 > 0:
            tempt = tempt/10
        label = tempt
        fp.write(value+str(label)+"\n")
        if i % 10000 == 0:
            print i
    fp.close()
