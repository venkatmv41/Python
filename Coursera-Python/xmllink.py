import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter location: ')
x = 1
while(x>0):
    if len(url) < 1: break
    uh = urllib.request.urlopen(url, context=ctx)
    data = uh.read()
    z = data.decode()
    tree = ET.fromstring(z)
    results = tree.findall('comments/comment')
    array = []
    for item in results:
        y = item.find('count').text
        w = int(y)
        array.append(w)
    sum = sum(array)
    print(sum)
    x =x-1
