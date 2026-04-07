import urllib.request
import urllib.parse
import json
import re
import io
import PyPDF2

def search_core():
    query = urllib.parse.quote('("Acculturative Stress Scale for International Students" OR "ASSIS") "homesickness" "guilt"')
    # Core API requires key usually, let's try Semantic Scholar instead
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,url,openAccessPdf&limit=10"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            for item in data.get('data', []):
                pdf_info = item.get('openAccessPdf')
                if pdf_info and pdf_info.get('url'):
                    pdf_url = pdf_info['url']
                    print("Found PDF:", pdf_url)
                    try:
                        pdf_req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
                        pdf_data = urllib.request.urlopen(pdf_req, timeout=10).read()
                        reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
                        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                        
                        # Look for strings that look like items
                        print("Length of text:", len(text))
                        res = re.findall(r'(?:\d{1,2}\.|\b)Homesickness\b.*?(?=\d{1,2}\.|\n|$)', text, re.IGNORECASE)
                        if res:
                            print("Matches found in", pdf_url)
                            # Let's save the text to a file so we can analyze it
                            with open('pdf_content.txt', 'w', encoding='utf-8') as f:
                                f.write(text)
                            return
                    except Exception as e:
                        print("Error reading PDF:", e)
    except Exception as e:
        print("Error fetching from Semantic Scholar:", e)

search_core()
