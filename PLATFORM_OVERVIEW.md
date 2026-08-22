# FeedbackForge: Platform Architecture & AI Methodologies

**FeedbackForge** is a production-ready, enterprise-grade AI Intelligence platform designed to transform raw customer feedback into disruptive business strategy. 

---

## 🚀 AI Agent Orchestration (The Pipeline)
The platform uses **LangGraph** to build a stateful, multi-agent workflow that processes every review through a specialized pipeline:

1.  **Triage Agent (Zero-Shot Classifier)**: Categorizes reviews into *Service, Product, App,* or *Atmosphere* using immediate semantic classification.
2.  **Prioritization Agent (Heuristic Scorer)**: A hybrid agent that uses a deterministic algorithm to calculate an **Urgency Score** based on numerical rating and textual density.
3.  **Synthesizer Agent (RAG Clusterer)**: Performs **Retrieval-Augmented Generation** to query the ChromaDB vector store for similar historical patterns, identifying the true "Root Cause" of the current issue.
4.  **Forge Agent (Generative Strategist)**: Translates technical root causes into actionable business output (e.g., PRD snippets for Product issues or Operational Directives for Service failures).

---

## 1. Data Ingestion & Intelligence Pipeline
When a CSV of customer reviews is uploaded, the backend triggers a multi-stage **Intelligence Pipeline**:

### 🧠 Dual-Stage NLP Analysis
- **Coarse Sentiment**: Each review is processed by **VADER** to determine overall polarity (Positive, Neutral, Negative).
- **Fine-Grained Emotion**: A **DistilRoBERTa Transformer** model performs multi-label classification to extract specific emotional triggers like *Anger, Frustration, Disappointment, and Joy*.

### 🌐 Vectorization & Storage
- **Embeddings**: Reviews are converted into high-dimensional vectors using the `all-MiniLM-L6-v2` model.
- **Vector Database**: Results are stored in **ChromaDB**, enabling semantic search and context-aware retrieval.

### 🕸️ GraphRAG & Ontology Extraction
- The system uses **spaCy** and **NetworkX** to build an interactive **Knowledge Graph**.
- **Methodology**: It extracts noun-adjective co-occurrence pairs (e.g., "Service" is frequently connected to "Slow") to map the customer's mental model of the product.

---

## 🧠 Core AI Methodologies
- **Stateful Multi-Agent Systems (MAS)**: Leveraging LangGraph to ensure each step of the analysis is handled by a specialized persona with its own prompt engineering and temperature settings.
- **RAG (Retrieval-Augmented Generation)**: Grounding the Groq/Llama 3.3 models in real-world historical data to prevent hallucinations and provide data-driven advice.
- **Few-Shot & Zero-Shot Learning**: Used across triage and synthesis to identify complex business patterns without requiring large-scale training sets.
- **Multi-Label Emotion Modeling**: Utilizing **DistilRoBERTa** transformers to detect simultaneous emotional states (e.g., a customer feeling both "Anger" and "Disappointment").
- **Heuristic-Driven Prioritization**: Combining LLM reasoning with mathematical scoring to ensure consistent emergency detection.

---

## 2. Core Dashboard Functional Workspaces

### 📊 Data Ingestion Dashboard
- **Sentiment Topology**: Visual breakdown of global customer mood.
- **Emotion Trajectory Intelligence**: A longitudinal graph tracking how specific feelings (Anger, Disappointment) evolve over time, detecting systematic risks.
- **Ontology Map**: Interactive force-directed graph showing the relationship between noun-adjectives in the dataset.

### 🛠️ Agent Portal
- **Real-time Analysis**: Instant classification and root cause analysis for single reviews.
- **Predictive Autocomplete**: A local **GPT-model (`distilgpt2`)** predicts text in real-time to assist agents in drafting responses.

### 🧠 Owner Insights
- **Systemic Issue Detection**: Aggregates the "worst" review clusters into specialized business insights.
- **Strategic Frameworks**: Generates highly disruptive, non-generic solution frameworks (e.g., "The Digital Siege").

### ⚡ Executive Advisor (Executive SLM)
- **Emotion-Aware RAG**: Chat interface that retrieves relevant reviews *with their emotional metadata*.
- **Strategic Starters**: AI-generated, data-driven questions based on your specific historical data.

---

## 3. Technology Stack
- **Frontend**: React (Vite) + Tailwind CSS + Recharts + React Force Graph.
- **Backend**: FastAPI (Python) + Uvicorn.
- **AI Core**: **Groq (Llama 3.3)** for strategic reasoning + **HuggingFace Transformers** for local NLP.
- **Data Layer**: **ChromaDB** (Vector) + **Pandas** (Dataframes) + **spaCy** (NER/POS).

---

## 4. End-to-End Flow Summary
1. **INPUT**: Raw review data (CSV).
2. **PROCESS**: Extract sentiment/emotion, generate embeddings, and build co-occurrence maps.
3. **SYNTHESIZE**: Aggregate problematic review clusters into strategic insights via Llama 3.3.
4. **INTERACT**: Engage with an AI Advisor that has full emotional and historical context of the entire business ecosystem.
