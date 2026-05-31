import urllib.request
import xml.etree.ElementTree as ET

def search_arxiv():
    url = 'http://export.arxiv.org/api/query?search_query=all:%22Levine%22+AND+all:%22hat%22&max_results=10'
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        
        # Define namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        entries = root.findall('atom:entry', namespaces)
        print(f"Found {len(entries)} entries:")
        for entry in entries:
            title = entry.find('atom:title', namespaces).text.strip()
            published = entry.find('atom:published', namespaces).text.strip()
            summary = entry.find('atom:summary', namespaces).text.strip()
            id_url = entry.find('atom:id', namespaces).text.strip()
            
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name = author.find('atom:name', namespaces).text.strip()
                authors.append(name)
                
            print(f"Title: {title}")
            print(f"Authors: {', '.join(authors)}")
            print(f"Published: {published}")
            print(f"URL: {id_url}")
            print(f"Summary: {summary}")
            print("-" * 40)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    search_arxiv()
