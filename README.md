# G-R-A-P-H-I-T-Y-Graph-Processing-Database-System
```markdown

Graphity is a powerful Graph Processing Database System designed to provide a web-based interface for querying, manipulating, and retrieving various aspects of graphs, as well as performing operations on them. Traditional query languages like SQL are not ideally suited for graph-related tasks, which is where Graphity comes in, leveraging a combination of graph algorithms to facilitate efficient graph management within a database context.

## Repository

Find the source code on GitHub: [Graphity Repository](https://github.com/Sukhomay/G-R-A-P-H-I-T-Y-Graph-Processing-Database-System)

## Introduction

Graphity offers a versatile platform for working with various types of graphs. Some of its prospective use cases include:

- **Social Networks Analysis:** Modeling social networks where nodes represent users and edges represent relationships.
- **Fraud Detection:** Detecting fraudulent activities by analyzing patterns of connections between entities such as accounts, transactions, and IP addresses.
- **Knowledge Graphs:** Building structured knowledge graphs representing relationships between concepts, entities, and attributes.
- **Biological and Medical Research:** Modeling complex biological networks like protein-protein interactions, genetic pathways, and disease associations.
- **Geospatial Analysis:** Analyzing geospatial data such as transportation networks, supply chains, and location-based services.
- **Semantic Web:** Representing linked data and ontologies in the Semantic Web.

## Functions [Objectives]

Graphity supports various fundamental operations including:

- Metadata retrieval of the graph
- Counting nodes and edges
- Counting Weakly and Strongly Connected Components (WCC and SCC)
- Determining the size of Largest Strongly and Weakly Connected Components
- Counting simple cycles
- Time and space analysis for pre-processing
- Retrieving indegree and outdegree of queried nodes
- Calculating PageRank value and rank of queried nodes
- Finding the shortest distance between two queried nodes
- Discovering K-nearest neighbors of queried nodes
- Generating rank lists of nodes within a queried range
- Determining whether queried nodes belong to the same WCC or SCC

## Salient Features

Graphity boasts several notable features, including:

- Emulation of block access from disk through binary files of fixed size (512B) for efficient performance monitoring.
- Implementation of primary clustering ordered index for quick access to variable-sized adjacency lists.
- Storage of global graph information in a single file Metadata.txt.
- Storage of node information in fixed-size chunks of 64 B, improving performance on node-specific queries.
- Use of static hashing for storing PageRank-Lists to handle range queries efficiently.
- Customized adjacency list storage for optimized memory usage and support for both weighted and unweighted graphs.
- Web-based interface for graph manipulation, providing accessibility from any device, intuitive usability, and seamless integration with other web services, enhancing collaboration, scalability, and reducing deployment complexity.

---
This README provides an overview of Graphity's capabilities and features. For detailed usage instructions and documentation, please refer to the project repository.
```

Feel free to tweak or expand upon any section as needed!
