import re
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter - ')
html = urllib.request.urlopen(url, context=ctx).read()
soup = BeautifulSoup(html, 'html.parser')
#print(soup.prettify())
# Retrieve all of the anchor tags
tags = soup('span')
array =[]
for tag in tags:
    x = tag.decode().strip()
    z = re.findall("[0-9]*", x)
    integer = [array.append(i) for i in z if(i.strip())]
converted = list(map(int,array))
sum = sum(converted)
print(sum)