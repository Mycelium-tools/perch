import json
import requests
import argparse
from osfclient.api import OSF
from urllib.parse import quote

def search_osf_by_tag(tag, construct_meta=False):
    """
    Searches OSF projects by a single tag.
    Returns either a structured RAG metadata format or a full OSF attribute dump.
    """
    encoded_tag = quote(tag)
    search_url = f"https://api.osf.io/v2/nodes/?filter[tags]={encoded_tag}"
    
    print(f"🔍 Searching OSF for projects tagged with: '{tag}'...")
    
    try:
        response = requests.get(search_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching from OSF API: {e}")
        return []

    search_data = response.json()
    osf = OSF()
    osf_results = []

    for item in search_data.get('data', []):
        node_id = item['id']
        attrs = item.get('attributes', {})
        
        try:
            project = osf.project(node_id)
            
            # Fetch Contributors
            contrib_url = f"https://api.osf.io/v2/nodes/{node_id}/contributors/?embed=users"
            c_res = requests.get(contrib_url)
            c_data = c_res.json()
            
            names = []
            for c in c_data.get('data', []):
                user_attr = c.get('embeds', {}).get('users', {}).get('data', {}).get('attributes', {})
                name = user_attr.get('full_name')
                if name:
                    names.append(name)
            
            contributor_str = ", ".join(names) if names else "OSF Contributor"

            if construct_meta:
                # OPTION A: Structured RAG Metadata Proposal with default values
                raw_date = attrs.get('date_modified') or attrs.get('date_created', "")
                osf_results.append({
                    "type": "url",
                    "source": f"https://osf.io/{node_id}/",
                    "namespace": "animal_policies",
                    "meta": {
                        "name": project.title,
                        "url": f"https://osf.io/{node_id}/",
                        "organization": contributor_str,
                        "primary_focus": "Effective Advocacy", # Default value
                        "pub_date": raw_date[:10] if raw_date else "N/A",
                        "doc_type": "report", # Default value
                        "tags": attrs.get('tags', [])
                    }
                })
            else:
                # OPTION B: Raw OSF Attribute Dump
                osf_results.append({
                    "osf_id": node_id,
                    "title": project.title,
                    "description": project.description,
                    "contributors": names,
                    "url": item.get('links', {}).get('html'),
                    "tags": attrs.get('tags', []),
                    "date_created": attrs.get('date_created'),
                    "date_modified": attrs.get('date_modified'),
                    "public": attrs.get('public'),
                    "category": attrs.get('category')
                })
            
            print(f"✅ Captured: {project.title}")
            
        except Exception as e:
            print(f"⚠️ Skipping node {node_id} due to error: {e}")

    return osf_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSF Tag Search")
    parser.add_argument("tag", help="Tag to search for")
    parser.add_argument("--output", default="osf_results.json", help="Output filename")
    parser.add_argument(
        "--construct_meta", 
        action="store_true", 
        help="Flag to output structured RAG metadata schema. If false, dumps raw OSF attributes."
    )
    
    args = parser.parse_args()

    results = search_osf_by_tag(args.tag, construct_meta=args.construct_meta)
    
    if results:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n✨ Successfully saved {len(results)} results to {args.output}")