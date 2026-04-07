import urllib.request
import re

try:
    url = "https://html.duckduckgo.com/html/?q=site:core.ac.uk+\"Acculturative+Stress+Scale+for+International+Students\"+\"Homesickness\""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req).read().decode("utf-8")
    
    # Just grab plain snippets from duckduckgo
    snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.I | re.S)
    for s in snippets:
        print("Snippet:", s.strip())
        print("---")
except Exception as e:
    print(e)
