# test url for this code is http://py4e-data.dr-chuck.net/comments_1279644.json




import urllib.request
import json

url = input("enter url:  ")
linkd = urllib.request.urlopen(url).read()
data = json.loads(linkd)
array =[]
comm = data["comments"]
array = []
for x in comm:
    z =x["count"]
    w = array.append(z)
sum =sum(array)
print(sum)
