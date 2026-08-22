import spacy
import networkx as nx
import json

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: en_core_web_sm not found. Run 'python -m spacy download en_core_web_sm'.")
    # Graceful degradation if model isn't downloaded yet.
    nlp = None

kg = nx.Graph()

def extract_graph_from_df(df):
    """
    Parses reviews to find Noun -> Adjective relationships.
    Builds a Knowledge Graph. Capped at 200 reviews for speed.
    """
    global kg
    kg.clear()
    
    if nlp is None:
        return {"nodes": [], "links": []}
    
    node_counts = {}
    edge_counts = {}
    
    # Sample up to 1000 reviews for a detailed, high-density Knowledge Graph
    sample_df = df.head(1000)
    
    for _, row in sample_df.iterrows():
        review_text = row.get('Review', row.get('review', row.get('review_text', row.get('text', ''))))
            
        if not review_text or review_text.strip() in ["", "No Review Text"]:
            continue
        
        # Truncate to first 200 chars for speed
        doc = nlp(review_text.lower()[:200])
        
        # Link NOUNs to their descriptive ADJECTIVES
        for token in doc:
            if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                noun = token.head.lemma_
                adj = token.lemma_
                if len(noun) < 3 or len(adj) < 3:
                    continue
                node_counts[noun] = node_counts.get(noun, 0) + 1
                node_counts[adj] = node_counts.get(adj, 0) + 1
                edge = tuple(sorted([noun, adj]))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
                
            elif token.dep_ == "amod" and token.head.pos_ == "NOUN":
                noun = token.head.lemma_
                adj = token.lemma_
                if len(noun) < 3 or len(adj) < 3:
                    continue
                node_counts[noun] = node_counts.get(noun, 0) + 1
                node_counts[adj] = node_counts.get(adj, 0) + 1
                edge = tuple(sorted([noun, adj]))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

    # Top 150 edges for a rich, expansive graph display
    sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:150]
    
    for (source, target), weight in sorted_edges:
        if weight >= 2:  # At least 2 co-occurrences
            kg.add_node(source, val=node_counts[source], group="noun")
            kg.add_node(target, val=node_counts[target], group="adjective")
            kg.add_edge(source, target, weight=weight)
            
    return get_graph_data()

def get_graph_data():
    """Serializes the NetworkX graph to react-force-graph format."""
    nodes = [{"id": node, "val": kg.nodes[node].get("val", 1), "group": kg.nodes[node].get("group", "unknown")} for node in kg.nodes()]
    links = [{"source": u, "target": v, "weight": d.get("weight", 1)} for u, v, d in kg.edges(data=True)]
    return {"nodes": nodes, "links": links}
