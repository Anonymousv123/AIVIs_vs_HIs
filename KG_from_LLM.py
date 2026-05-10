from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field  
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

try:
  import google.colab
  from google.colab import output
  output.enable_custom_widget_manager()
except:
  pass

import pandas as pd
import os

# Colab secret keys
from google.colab import userdata

# Drive
from google.colab import drive
drive.mount('/content/drive')

OPENAI_API_KEY = userdata.get('api_key_openAI')

# alonlonn
NEO4J_USERNAME='neo4j'
NEO4J_DATABASE='neo4j'
NEO4J_URI='...'
NEO4J_PASSWORD='...'

PTH ='...'

"""# BUILD A KG FROM DOCUMENTS"""

# Open the Google Sheet by URL
df = pd.read_csv(os.path.join(PTH,'data/AIVI_HI.csv'))
df['id'] = range(1, len(df) + 1)
df = df[df['posts.comments.text'].notna()]

docs = df['posts.comments.text'].to_list()

len(docs)

"""## Create text chunks"""

def split_form10k_data_from_file(df):
    chunks_with_metadata = [] 
    for i in range(0,len(df)):
        chunks_with_metadata.append({
            'text': df.iloc[i]['posts.comments.text'],
            # metadata from looping...
            'author': df.iloc[i]['posts.comments.user'],
            'comment_likes': df.iloc[i]['pot.comment.likes_count'],
            'emoji': df.iloc[i]['emoji.comments.text'],
            'user_type': df.iloc[i]['user_type'],
            'created_utc': df.iloc[i]['posts.time'],
            'id': df.iloc[i]['id'],
        })
    return chunks_with_metadata

first_file_chunks = split_form10k_data_from_file(df)

"""## Create graph nodes from text chunks"""

merge_chunk_node_query = """
MERGE(mergedChunk:Chunk {chunkId: $chunkParam.id})
    ON CREATE SET
        mergedChunk.text = $chunkParam.text,
        mergedChunk.author = $chunkParam.author,
        mergedChunk.comment_likes = $chunkParam.comment_likes,
        mergedChunk.emoji = $chunkParam.emoji,
        mergedChunk.user_type = $chunkParam.user_type,
        mergedChunk.created_utc = $chunkParam.created_utc
RETURN mergedChunk
"""

# Set up connection to graph instance using LangChain
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

first_file_chunks[0]

# Create a single chunk node for now
kg.query(merge_chunk_node_query,
         params={'chunkParam':first_file_chunks[0], 'id': first_file_chunks[0]['id']})

kg.query("""
CREATE CONSTRAINT unique_chunk IF NOT EXISTS
    FOR (c:Chunk) REQUIRE c.id IS UNIQUE
""")

kg.query("SHOW INDEXES")

# create nodes for all chunks

node_count = 0
for chunk in first_file_chunks:
    print(f"Creating `:Chunk` node for chunk ID {chunk['id']}")
    kg.query(merge_chunk_node_query,
            params={
                'chunkParam': chunk
            })
    node_count += 1
print(f"Created {node_count} nodes")

# query to the graph to check how many nodes were created
kg.query("""
         MATCH (n)
         RETURN count(n) as nodeCount
         """)

"""## Create a vector index of text embeddings"""

kg.query("""
         CREATE VECTOR INDEX `form_10k_chunks` IF NOT EXISTS
          FOR (c:Chunk) ON (c.textEmbedding)
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
         }}
""")

kg.query("SHOW INDEXES")

"""### Calculate embedding vectors for chunks and populate index"""

kg.query("""
    MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
    WITH chunk, genai.vector.encode(
      chunk.text,
      "OpenAI",
      {
        token: $openAiApiKey,
        endpoint: 'YOUR_OPENAI_ENDPOINT'
      }) AS vector
    CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
    """,
    params={"openAiApiKey":OPENAI_API_KEY} )

"""# KG: entity recognotion and relation recognition

## load the data
"""

df.head()

# # connect to Neo4j graph DB

graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

from langchain_core.documents import Document

documents = []
for index, row in df.iterrows():
    doc = Document(
      page_content= row['selftext'],
      metadata={
           'author' : row['author'],
           'title' : row['title'],
           'created_utc' : row['created_utc'],
           'id' : row['id'],
           'num_comments' : row['num_comments'],
           'score' : row['score'],
           'subreddit' : row['subreddit'],
           'author_created_utc' : row['author_created_utc'],
           'ptsd' : row['ptsd']
      }
    )
    documents.append(doc)

from langchain_core.documents import Document

documents = []
for index, row in df.iterrows():
    doc = Document(
      page_content= row['cb_delivery_narrative'],
      metadata={
           'spcl5_total' : row['spcl5_total'],
           'ptsd' : row['y'],
           'author':"new_node",
           'id' : row["record_id"]
      }
    )
    documents.append(doc)

len(documents)

# Storing to graph database in NEO4J
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)



from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# # connect to Neo4j graph DB and read the graph
graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)


llm = ChatOpenAI(temperature=0,
                 model="gpt-4o",  # Updated to gpt-4o
                 openai_api_key = OPENAI_API_KEY
                 )

chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True)

# Initialize the Neo4j driver

driver = GraphDatabase.driver(
        uri = NEO4J_URI,
        auth = (NEO4J_USERNAME,
                NEO4J_PASSWORD)
        )

# count number of nodes and edges in NEO4J

def count_nodes_and_edges(driver):
    # Define Cypher queries
    node_query = "MATCH (n) RETURN COUNT(n) AS NumberOfNodes"
    edge_query = "MATCH ()-[r]->() RETURN COUNT(r) AS NumberOfEdges"

    with driver.session() as session:
        # Execute the queries and fetch results
        node_result = session.run(node_query)
        edge_result = session.run(edge_query)

        # Extract counts from results
        number_of_nodes = node_result.single()['NumberOfNodes']
        number_of_edges = edge_result.single()['NumberOfEdges']

        # Print the results
        print(f"Number of Nodes: {number_of_nodes}")
        print(f"Number of Edges: {number_of_edges}")

# Call the function
count_nodes_and_edges(driver)

"""### query the graph using LLM"""

# vector search (RAG)?

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(model='text-embedding-3-large',dimensions=1536, openai_api_key = OPENAI_API_KEY),
    search_type="hybrid",
    node_label="Document",
    url = NEO4J_URI,
    username = NEO4J_USERNAME,
    password = NEO4J_PASSWORD,
    database = NEO4J_DATABASE,
    text_node_properties=["text"],
    embedding_node_property="embedding"
)


import pandas as pd
from neo4j import GraphDatabase

# Load from environment
NEO4J_URI = userdata.get('NEO4J_URI')
NEO4J_DATABASE = userdata.get('NEO4J_DATABASE')
NEO4J_PASSWORD = userdata.get('NEO4J_PASSWORD')
NEO4J_USERNAME = userdata.get('NEO4J_USERNAME')

OPENAI_API_KEY = userdata.get('api_key_openAI')

# Create an auth tuple using previously fetched values
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

with GraphDatabase.driver(NEO4J_URI, auth=AUTH) as driver:
    driver.verify_connectivity()

# Load the edge list from the file
file_path = '/.../KG_cb-ptsd - Sheet1.csv'
edges = pd.read_csv(file_path)

edges

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Function to create nodes
def create_nodes(tx, nodes):
    query = """
    UNWIND $nodes AS node
    MERGE (n:Entity {name: node})
    """
    tx.run(query, nodes=nodes)

# Function to create edges with relationship type
def create_edges(tx, edges):
    query = """
    UNWIND $edges AS edge
    MATCH (a:Entity {name: edge.Entity_A})
    MATCH (b:Entity {name: edge.Entity_B})
    MERGE (a)-[r:`RELATION` {type: edge.Relation}]->(b)
    """
    tx.run(query, edges=edges)

# Push the data to Neo4j
with driver.session() as session:
    # Create nodes
    nodes = pd.concat([edges['Entity_A'], edges['Entity_B']]).unique().tolist()
    session.write_transaction(create_nodes, nodes)

    # Create edges
    edge_list = edges.to_dict(orient="records")  # Convert to a list of dictionaries
    session.write_transaction(create_edges, edge_list)

print("Graph successfully pushed to Neo4j!")

# Close the Neo4j connection
driver.close()
