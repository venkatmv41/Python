# link ____ :   http://py4e-data.dr-chuck.net/known_by_Holly.html



import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

count = int(input(" enter count :  "))
position = int(input(" enter pos:  "))
url = input('Enter - ')

while(count>0):
    html = urllib.request.urlopen(url, context=ctx).read()
    soup = BeautifulSoup(html, 'html.parser')
    # Retrieve all of the anchor tags
    tags = soup('a')
    array = []
    for tag in tags:
        x = tag.get('href', None)
        array.append(x)
    z = array[position-1]
    url =z
    print(url)
    count-=1



