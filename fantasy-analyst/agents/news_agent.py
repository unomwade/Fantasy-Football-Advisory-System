"""
News Agent

Answers: "What injuries, lineup decisions, and/or off-field considerations
are affecting this player?"

Pipeline:
  1. fetch_espn_news()       -- pull recent articles from ESPN's public API
  2. build_news_documents()  -- normalize articles into ChromaDB-ready docs
  3. refresh_news()          -- embed + upsert into ChromaDB (TTL-gated)
  4. ask_news_agent()        -- RAG-backed Q&A over the news collection

refresh_news() is TTL-gated so repeated calls within the same session don't
re-fetch and re-embed unnecessarily. Call session_startup() (see startup.py)
once per session rather than calling refresh_news() directly in app code.
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone

import chromadb
import requests
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------------------------

CHROMA_PATH = os.environ.get("CHROMA_PATH", "/app/data/chroma")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
news_collection = chroma_client.get_or_create_collection(
    name="nfl_news",
    metadata={"hnsw:space": "cosine"}
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# ESPN fetcher
# ---------------------------------------------------------------------------

ESPN_NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"


def fetch_espn_news(limit: int = 100) -> list[dict]:
    """
    Fetch recent NFL news articles from ESPN's unofficial API.
    No API key required.

    Returns a list of dicts with keys:
      headline, description, published, players_mentioned, teams_mentioned, url
    """
    all_articles = []
    page = 1
    per_page = 25  # ESPN's max per request

    while len(all_articles) < limit:
        params = {"limit": per_page, "page": page}
        try:
            r = requests.get(ESPN_NEWS_URL, params=params, timeout=10)
            r.raise_for_status()
        except requests.RequestException:
            break

        data = r.json()
        articles = data.get("articles", [])
        if not articles:
            break

        for article in articles:
            categories = article.get("categories", [])
            players_mentioned = [
                cat.get("athlete", {}).get("displayName", "")
                for cat in categories
                if cat.get("type") == "athlete" and cat.get("athlete")
            ]
            teams_mentioned = [
                cat.get("team", {}).get("abbreviation", "")
                for cat in categories
                if cat.get("type") == "team" and cat.get("team")
            ]

            all_articles.append({
                "headline": article.get("headline", ""),
                "description": article.get("description", ""),
                "published": article.get("published", ""),
                "players_mentioned": players_mentioned,
                "teams_mentioned": teams_mentioned,
                "url": article.get("links", {}).get("web", {}).get("href", ""),
            })

        if len(articles) < per_page:
            break
        page += 1

    return all_articles[:limit]


# ---------------------------------------------------------------------------
# Staleness config
# ---------------------------------------------------------------------------

NEWS_TTL_SECONDS = 3 * 3600  # treat cache as stale after 3 hours

_news_last_refreshed_at: float | None = None  # epoch seconds (UTC)


def _news_cache_is_fresh() -> bool:
    """True if the collection was refreshed recently enough to skip ESPN."""
    if _news_last_refreshed_at is None:
        return False
    return (time.time() - _news_last_refreshed_at) < NEWS_TTL_SECONDS


def build_news_documents(espn_articles: list[dict]) -> list[dict]:
    """Convert raw ESPN articles into ChromaDB-ready documents."""
    documents = []

    for article in espn_articles:
        if not article["headline"] and not article["description"]:
            continue

        text_parts = [f"Headline: {article['headline']}"]

        if article["published"]:
            text_parts.append(f"Published: {article['published']}")
        if article["teams_mentioned"]:
            text_parts.append(f"Teams: {', '.join(article['teams_mentioned'])}")
        if article["players_mentioned"]:
            text_parts.append(f"Players mentioned: {', '.join(article['players_mentioned'])}")
        if article["description"]:
            text_parts.append(f"Summary: {article['description']}")
        if article["url"]:
            text_parts.append(f"Source: {article['url']}")

        text = "\n".join(text_parts)
        doc_id = hashlib.md5(
            (article["url"] or article["headline"]).encode()
        ).hexdigest()

        documents.append({
            "id": doc_id,
            "text": text,
            "metadata": {
                "headline": article["headline"][:200],
                "published": article["published"],
                "players": ", ".join(article["players_mentioned"])[:200],
                "teams": ", ".join(article["teams_mentioned"])[:100],
                "url": article["url"][:500],
                "source": "espn",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return documents


def refresh_news(espn_limit: int = 100, force: bool = False) -> dict:
    """
    Refresh the news pipeline -- fetch ESPN articles, embed them, and
    upsert into ChromaDB.

    Refresh is skipped when force=False AND the in-memory cache is still
    within NEWS_TTL_SECONDS (cold start always refreshes even if the
    collection has persisted docs from a previous container run, since
    stale persisted data is worse than a fresh fetch).

    Returns a small status dict so callers (e.g. the Streamlit UI) can
    display what happened without parsing log output.
    """
    global _news_last_refreshed_at

    if not force and _news_cache_is_fresh():
        return {
            "refreshed": False,
            "reason": "cache_fresh",
            "doc_count": news_collection.count(),
        }

    espn_articles = fetch_espn_news(limit=espn_limit)
    documents = build_news_documents(espn_articles)

    batch_size = 100
    upserted = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # Deduplicate within the batch -- ChromaDB rejects duplicate IDs
        # even within a single upsert call.
        seen = {}
        for d in batch:
            seen[d["id"]] = d
        batch = list(seen.values())

        texts = [d["text"] for d in batch]
        ids = [d["id"] for d in batch]
        metas = [d["metadata"] for d in batch]
        embeddings = embed_texts(texts)

        news_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas
        )
        upserted += len(batch)

    _news_last_refreshed_at = time.time()

    return {
        "refreshed": True,
        "articles_fetched": len(espn_articles),
        "docs_upserted": upserted,
        "doc_count": news_collection.count(),
    }


# ---------------------------------------------------------------------------
# RAG retrieval + agent
# ---------------------------------------------------------------------------

def retrieve_relevant_news(query: str, n_results: int = 5) -> str:
    """Retrieve relevant NFL news from ChromaDB, with source + date."""
    query_embedding = embed_texts([query])[0]

    results = news_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"][0]:
        return "No relevant news found."

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        relevance = round((1 - dist) * 100, 1)
        source = meta.get("source", "unknown")
        published = meta.get("published", "")[:10]
        output.append(
            f"[Relevance: {relevance}% | Source: {source} | Date: {published}]\n{doc}"
        )

    return "\n\n---\n\n".join(output)


news_tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_relevant_news",
            "description": (
                "Search the NFL news vector store for injury reports and player updates. "
                "Pass a natural language query such as a player name or topic. "
                "You may call this multiple times with different queries to improve coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query, e.g. player name or injury topic."
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

NEWS_SYSTEM_PROMPT = """
You are an NFL injury and news analyst for fantasy football. Your job is to
report all news relevant to the player(s) given the context of the question.

For example, if the question was "Is Saquon Barkley going to play this week?"
Then return information about his availability.
But if asked "Should I start Christian McCaffrey or Jahmyr Gibbs?"
You would want to look at not only their news/injuries, but also at their
quarterbacks, opposing defenses, etc.
Anything that might have an affect on their fantasy score.

You have access to a vector store of current NFL player news and injury reports
via the retrieve_relevant_news tool. All documents come from ESPN news articles.

Rules:
- Always call retrieve_relevant_news before answering any question about a player.
- You may call it multiple times with different queries if the first returns weak results
  (e.g. search by player name, then by team name, then by injury type).
- Always note the publication date of your source -- older articles may be stale.
- Report injury status clearly: Active, Questionable, Doubtful, Out, IR.
- If no news is found, say so explicitly.
- Be concise -- focus on what matters for fantasy decisions.
"""


def ask_news_agent(user_question: str) -> str:
    """Answer a fantasy football news/injury question via RAG over ESPN articles."""
    messages = [
        {"role": "system", "content": NEWS_SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=news_tools,
        tool_choice="required"
    )
    message = response.choices[0].message

    max_iterations = 6
    iteration = 0

    while message.tool_calls and iteration < max_iterations:
        iteration += 1
        messages.append(message)

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "retrieve_relevant_news":
                tool_response = retrieve_relevant_news(
                    query=arguments["query"],
                    n_results=arguments.get("n_results", 5)
                )
            else:
                tool_response = f"Unknown tool: {function_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_response
            })

        next_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=news_tools,
            tool_choice="auto"
        )
        message = next_response.choices[0].message

    return message.content
