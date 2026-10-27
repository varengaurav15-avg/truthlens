from pathlib import Path
import pandas as pd

def build_graph_from_triples(csv_path: Path, neo4j_uri: str, user: str, password: str):
    from neo4j import GraphDatabase
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    if len(cols) >= 3:
        s, r, o = cols[0], cols[1], cols[2]
    else:
        s, r, o = cols[0], "relation", "object"
    if "subject" in df.columns and "relation" in df.columns and "object" in df.columns:
        s, r, o = "subject", "relation", "object"
    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        for _, row in df.iterrows():
            sub = str(row.get(s, "unknown"))
            rel = str(row.get(r, "relation"))
            obj = str(row.get(o, "object"))
            safe_rel = rel.replace(" ", "_").replace("`", "") or "RELATED_TO"
            query = f"MERGE (a:Entity {{name: $s}}) MERGE (b:Entity {{name: $o}}) MERGE (a)-[r:`{safe_rel}`]->(b)"
            session.run(query, s=sub, o=obj)
    driver.close()