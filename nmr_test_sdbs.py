import requests
from bs4 import BeautifulSoup

url = "https://sdbs.db.aist.go.jp/CompoundLanding.aspx?sdbsno=19672"

r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
print(r.status_code)
print(r.url)
print(r.text[:1000])

soup = BeautifulSoup(r.text, "html.parser")
text = soup.get_text(" ", strip=True)

print(text[:3000])
print("Contains 1H NMR?", "1H NMR" in text or "H NMR" in text)