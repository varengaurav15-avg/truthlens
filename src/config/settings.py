from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    hf_token: str = ""
    model_cache_dir: str = "./models"
    claim_model: str = "bert-base-uncased"
    deepfake_model: str = "efficientnet_b0"

    # Live evidence retrieval. Set enable_wiki_evidence=false in .env to run
    # fully offline against the curated fact file only.
    enable_wiki_evidence: bool = True
    # 6s was too tight: large Wikipedia extracts timed out, and a timeout is
    # indistinguishable from "no evidence" to the adjudicator. Payloads are now
    # capped via exchars, but keep real headroom.
    wiki_timeout: float = 15.0

    # Optional LLM adjudication over retrieved evidence.
    #   llm_backend: auto | ollama | transformers | none
    # "auto" uses Ollama when it is running and silently skips otherwise, so
    # the pipeline never breaks just because no LLM is installed.
    #   llm_provider: auto | groq | gemini | openai | ollama | none
    # "auto" picks the best available: a cloud provider if an API key is set,
    # otherwise a local Ollama install, otherwise nothing (NLI-only).
    #
    # Accuracy scales hard with model size. Groq's free tier serves
    # Llama-3.3-70B — roughly 20x the parameters of a local 3B model — and is
    # the single biggest quality win available here.
    llm_provider: str = "auto"
    llm_api_key: str = ""
    llm_model: str = ""            # blank = provider default
    llm_base_url: str = ""         # blank = provider default
    llm_timeout: float = 45.0
    llm_max_claims: int = 8        # bounds latency on long articles

    # legacy alias, still honoured
    llm_backend: str = "auto"

settings = Settings()