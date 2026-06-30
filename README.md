# LangGraph Fan-Out, Guardrails, and pgvector RAG

[![CI](https://github.com/sheldonlsides/langgraph-agent-guardrails/actions/workflows/ci.yml/badge.svg)](https://github.com/sheldonlsides/langgraph-agent-guardrails/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Tool-using agents are easy to demo and hard to make reliable. This single LangGraph notebook works through three patterns that hold up in production:

- **Fan-out / fan-in with `Send()`** and an `operator.add` reducer, joined at a barrier node that
  runs deterministic Python instead of trusting the LLM for arithmetic.
- **Layered prompt-injection defense.** A structured `ScopeCheck` gate refuses off-topic requests
  before any tool runs, a `GUARDRAILS` prefix sits on every agent's system prompt, and all web and
  tool output is treated as untrusted data.
- **pgvector RAG grounding** with `langchain-postgres` v2 (`PGVectorStore` + HNSW cosine index),
  embedded once and idempotently loaded on every subsequent run.

The use-case in this example is a meal planner. You will give it a request like *"a 1-day high-protein, low-carb plan, 1,800
to 2,000 kcal, include snacks, allergic to shellfish"* and a team of agents plans unique dishes,
researches a real recipe for each one in parallel, grounds every ingredient in a nutrition database,
and renders a calorie-checked meal plan as markdown. Every macro number is computed in Python, not
invented by the LLM.

Everything lives in one notebook: `src/agent_guardrails.ipynb`.

---

## What you get

A self-contained notebook that builds the graph end to end, and the patterns you can lift from it:

- **LangGraph fan-out / fan-in**: a `StateGraph` with dynamic parallelism via `Send(...)` and an
  `operator.add` reducer that concatenates worker results at a deterministic barrier node, plus a
  refusal short-circuit.
- **Scope-gating & prompt-injection defense**: a structured `ScopeCheck` gate plus a single
  `GUARDRAILS` block that refuses out-of-scope and injection-style requests *before* any tool runs,
  with all web/tool output treated as untrusted data.
- **RAG grounding with pgvector**: a retrieval tool over a real ~327K-row food database
  (OpenNutrition), embedded locally and queried with `langchain-postgres` v2 (`PGVectorStore` +
  HNSW cosine index).
- **Keeping the math out of the model**: structured Pydantic outputs plus deterministic Python for
  every number the user sees, rendered to markdown and written to disk at every step.
- **A provider-agnostic model factory**: one `create_model()` call swaps between Bedrock, OpenAI,
  and Anthropic via env vars, with no code changes.
- **A notebook that asserts its own claims**: the final cells run as a verification pass mapped to
  the [OWASP Top 10 for LLM Applications (2025)](https://genai.owasp.org/llm-top-10/). Four probes
  cover direct prompt injection (LLM01), indirect prompt injection via poisoned tool output (LLM01),
  fabricated data and threshold bypass (LLM09 Misinformation), and insecure inter-agent communication
  (LLM01 plus LLM05 Improper Output Handling). If the notebook runs end to end, the claims hold.

## Why this exists

As many of us know, LLMs are not great at math. If you ask an LLM to sum a column of macros across a week of meals, the
errors can compound silently. The same pattern shows up anywhere a tool-using agent has to roll up
numeric facts: invoices, financial summaries, capacity plans, anything where a small rounding error
becomes a large reporting error downstream.

The fix is to let the model plan and research, then let deterministic Python do the math. This
notebook shows that split end to end with a meal-planner as the use case. Ingredients are looked up in a
pgvector store that contains numerical nutrition information; portions are scaled by gram weight; meal, day, and grand
totals are sums in code. The model picks dishes, finds recipes, and estimates portion sizes.

The same notebook also demonstrates dynamic parallelism (one Recipe worker per meal slot, running
concurrently) and a layered defense against prompt injection. Those patterns are the reusable parts;
the meal planning agent is used to show them in action.

## How it works at a glance

```mermaid
flowchart TD
    START([request]) --> CHEF[chef_plan<br/>ScopeCheck + unique dish per slot]
    CHEF -->|out of scope| SUM[chef_summary]
    CHEF -->|in scope: Send fan-out| W1[recipe_worker 1]
    CHEF --> W2[recipe_worker 2]
    CHEF --> WN[recipe_worker N]
    W1 --> BAR[meal_planner barrier<br/>deterministic macros + render]
    W2 --> BAR
    WN --> BAR
    BAR --> SUM
    SUM --> END([END])
```

- **`chef_plan`** *(Chef, temp 0.7)*: runs a `ScopeCheck`; if the request isn't about food it
  refuses and the whole graph short-circuits. Otherwise it parses the day/meal/snack counts and any
  calorie range, then assigns a **unique dish to every slot** (e.g. *Day 1 Breakfast → Spinach &
  Feta Egg Scramble*).
- **`route_after_chef`**: the conditional edge. On refusal it routes straight to the summary; in
  scope it emits one `Send("recipe_worker", RecipeTask(...))` per planned meal (the fan-out).
- **`recipe_worker`** *(Recipe, temp 0)*: one parallel worker per slot. It web-searches for the
  exact assigned dish (Tavily), fetches the page, grounds each ingredient in pgvector, estimates
  portion grams toward the meal's calorie budget, and emits a structured `Recipe`. Workers append to
  `recipes: Annotated[list[Recipe], operator.add]` (the fan-in).
- **`meal_planner`** *(Planner, temp 0)*: the barrier; runs once all workers finish. It enforces
  per-day calorie caps by scaling portions (`_enforce_calories`, capped at `MAX_PORTION_SCALE = 2.5`),
  asks the model only for a title + one-sentence intro, then renders the markdown deterministically
  (`_render_plan_md`) and writes it to disk.
- **`chef_summary`**: closes the loop with the saved path and calorie range (or the refusal).

State lives in one `ChefState` Pydantic model passed through every node.

## Who this is for

You're comfortable with Python and have seen an LLM agent before; you don't need prior LangGraph
experience. The notebook builds the graph step by step. If `StateGraph`, nodes, and edges are new,
I recommend checking out the [LangGraph quickstart](https://langchain-ai.github.io/langgraph/) first.

This notebook is local-first. It needs a local PostgreSQL plus pgvector database. The Quick start below gets you running in a few commands.

## What's in the box

| Path | Purpose |
|------|---------|
| `src/agent_guardrails.ipynb` | The whole thing: vector store, tools, guardrails, agents, graph, and example runs, top to bottom. |
| `src/common/model_factory.py` | `create_model()`: provider-agnostic LangChain chat model (Bedrock / OpenAI / Anthropic), fail-fast on missing config. |
| `src/vectorstore/` | The pgvector backend (`build_or_load_pgvector`): idempotent build/load of the nutrition table. |
| `deploy/` | Docker Compose for PostgreSQL + pgvector (`pgvector/pgvector:pg17`). See `deploy/README.md`. |
| `docs/pgvector-setup.md` | pgvector tuning notes and the AWS RDS path. |
| `src/data/` | Where the OpenNutrition TSV lives (≈269 MB, **gitignored**; auto-fetched by `scripts/download_data.py`). |
| `.env.example` | Every environment variable, documented. Copy to `.env`. |

## Quick start (local)

Requires **Python 3.12**, the [uv](https://docs.astral.sh/uv/) package manager, and **Docker**
(with Compose) for the local PostgreSQL + pgvector database in step 3 (or bring your own; see
`docs/pgvector-setup.md`). Run every command below from the **project root**, and always through
`uv run` so the project `.venv` is used (not a system/Anaconda Python).

```bash
# 1. Install dependencies (from the committed lockfile)
uv sync                          # or: ./install_deps

# 2. Configure secrets. Copy the template and fill it in,
cp .env.example .env
#    set LLM_PROVIDER + LLM_PROVIDER_MODEL, the matching provider key,
#    TAVILY_API_KEY, and DATABASE_URL (see the table below)

# 3. Stand up PostgreSQL + pgvector (Docker)
cd deploy && docker compose up -d && cd ..   # details in deploy/README.md

# 4. Fetch the nutrition dataset (~60 MB zip → ≈269 MB TSV in src/data/)
#    The notebook auto-downloads it on first run; or fetch it now:
uv run python scripts/download_data.py        # idempotent: skips if already present

# 5. Run the notebook headless ...
uv run --with nbconvert -- jupyter nbconvert --to notebook --execute src/agent_guardrails.ipynb

#    ... or open it interactively
uv run --with nbconvert jupyter notebook src/agent_guardrails.ipynb
```

**First run is slow, once.** Building the vector table embeds the full ~326,759-food dataset on CPU
in batches of 5,000 and builds an HNSW index (this will take several minutes). Every run after that is
instant, a row-count check loads the existing table without re-embedding (drop the table to rebuild).

Generated plans land in `src/meal_plans/<slug>.md` (gitignored).

## Environment variables

Copy `.env.example` to `.env`. `create_model()` and the pgvector backend **fail fast** if their
required vars are missing.

| Variable | Required? | Used for | Notes |
|----------|-----------|----------|-------|
| `LLM_PROVIDER` | ✅ | Selects the LLM backend | `bedrock` \| `openai` \| `anthropic` |
| `LLM_PROVIDER_MODEL` | ✅ | Model id for that provider | e.g. `gpt-4o-mini`, `claude-sonnet-4-6` |
| `OPENAI_API_KEY` | if `openai` | OpenAI credentials | [Get a key](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | if `anthropic` | Anthropic credentials | [Get a key](https://console.anthropic.com/settings/keys); Bedrock uses AWS creds / `AWS_REGION` instead ([enable model access](https://console.aws.amazon.com/bedrock/home#/modelaccess)) |
| `TAVILY_API_KEY` | ✅ | Web recipe search | [Get a key](https://app.tavily.com/home); used by the `tavily_search` tool |
| `DATABASE_URL` | ✅ | pgvector connection | psycopg3 URL, e.g. `postgresql+psycopg://dev:devpass@localhost:5433/appdb` |
| `HF_TOKEN` | optional | Embedding model download | [Get a token](https://huggingface.co/settings/tokens); only needed for gated/private HF models |
| `USER_AGENT` | optional | Outbound HTTP header for page fetches | Defaults to `langgraph-agent-guardrails/1.0` |
| `LANGSMITH_API_KEY` / `LANGSMITH_ENDPOINT` / `LANGSMITH_PROJECT` | optional | LangSmith tracing | [Get a key](https://smith.langchain.com/settings) |

## Architecture notes

- **One state object, one reducer.** `ChefState` flows through every node. Its only field with a
  reducer is `recipes: Annotated[list[Recipe], operator.add]`, so N parallel workers each append one
  recipe and the results concatenate cleanly at the barrier. Every other field is plain
  last-writer-wins.
- **Fan-out is dynamic.** `route_after_chef` returns a *list* of `Send("recipe_worker", ...)` sized
  to the plan, so a 3-meal day spawns 3 workers and a 5-meal day spawns 5. No hardcoded width.
- **The scope gate runs first and blocks everything.** An out-of-scope request never reaches a tool,
  web search, or the filesystem: `chef_plan` returns a refusal and `route_after_chef` jumps to the
  summary. Defense-in-depth: `GUARDRAILS` is also prepended to every agent's system prompt, and
  Tavily results / fetched pages are treated as **data, never instructions**.
- **Macros are code, not model output.** `find_ingredients` returns per-100g facts from pgvector;
  `total_meal` scales by grams and sums; `_recompute_recipe_macros` and `_render_plan_md` produce
  the tables and totals. The model never adds two numbers the user sees.
- **pgvector, built once.** `build_or_load_pgvector` is idempotent via a **row-count gate**: a
  populated table is loaded as-is, never re-embedded. The HNSW cosine index is built *after* the
  bulk load (far cheaper than per-insert), and each row's stable OpenNutrition id is the primary
  key, so results stay traceable to source.
- **Per-agent temperatures, one factory.** `create_model()` is called three times: Chef at `0.7`
  (creative dish planning), Recipe and Planner at `0` (deterministic). The provider is chosen
  entirely by env vars.

## Common gotchas

- **Use `uv run` for everything.** Launching the notebook with a system/Anaconda kernel pulls in
  mismatched dependencies; the `uv run` commands above pin it to the project `.venv`.
- **`DATABASE_URL` is mandatory.** The nutrition store is pgvector-only. There is no local-file
  fallback. Stand up the DB with `deploy/` before running, and note the example port is **5433**
  (5432 is often a host-native Postgres).
- **The dataset isn't in the repo.** `opennutrition_foods.tsv` (~269 MB) exceeds GitHub's file-size
  limit and is gitignored. It's fetched from the official OpenNutrition source. The notebook
  auto-downloads it on first run, or run `uv run python scripts/download_data.py`. Both skip the
  download when the TSV is already in `src/data/`.
- **The first build really is slow.** Embedding ~327K foods on CPU takes many minutes. It's
  one-time; subsequent runs load instantly. Cap `MAX_ROWS` in the config cell for a fast dev subset.
- **Dietary constraints ride in the request.** Allergies and preferences (*"allergic to shellfish"*,
  *"vegetarian"*) go in the natural-language request. The Chef parses them when planning dishes.
- **Don't trust scraped pages.** Recipe pages and search results are extracted for facts only; the
  guardrails explicitly forbid following any instructions found inside them.
- **`torch==2.12.0` is hard-pinned.** This is intentional for reproducibility against the embedding
  model. If `uv sync` cannot resolve a torch wheel for your platform (older CUDA, certain Linux glibc
  versions, or specific Apple Silicon paths), install torch separately first matching your hardware,
  then re-run `uv sync`.

## Example output

A representative generated plan from a one-day high-protein request.

```markdown
# High-Protein, Low-Carb Day

A shellfish-free day built around lean protein and non-starchy vegetables.

> **Note:** Day 1 portions scaled ×1.12 to land inside the 1,800–2,000 kcal target.

## Day 1

### Day 1 Breakfast: Spinach & Feta Egg Scramble

A quick three-egg scramble with wilted spinach and a little feta.

| Ingredient | Amount | Calories | Protein (g) | Carbs (g) | Sugars (g) | Fat (g) |
|------------|--------|---------:|------------:|----------:|-----------:|--------:|
| Eggs       | 3 (150 g) | 215   | 18.8        | 1.1       | 1.1        | 14.9    |
| Spinach    | 60 g      | 14    | 1.7         | 2.2       | 0.3        | 0.2     |
| Feta       | 30 g      | 79    | 4.3         | 1.2       | 1.2        | 6.4     |

**Meal total:** calories=308, protein=24.8g, carbs=4.5g, fat=21.5g

**Recipe** - 10 min
1. Whisk the eggs...
2. ...

_Source: https://example.com/spinach-feta-scramble_

## Final Totals

| Day   | Calories | Carbs (g) | Protein (g) | Fat (g) |
|-------|---------:|----------:|------------:|--------:|
| Day 1 |    1,932 |        58 |         148 |     118 |
```

## Library pins

Python **3.12**, dependencies locked in `uv.lock`. `torch` is pinned (`torch==2.12.0` in
`pyproject.toml`) to a known-good combination for the embedding stack on macOS. Keep the pin unless
you've tested an upgrade.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities privately.

## License

Released under the [MIT License](LICENSE).

## Data & attribution

Nutrition data is from [OpenNutrition](https://www.opennutrition.app), licensed under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/) with contents under the
DbCL. The code only **fetches** the dataset from the official source at runtime (it is not
redistributed here), but it **displays** OpenNutrition data, so generated meal plans credit
OpenNutrition inline. Any database derived from it (e.g. the pgvector table) inherits ODbL
share-alike. The MIT license above covers this project's **code**, not the nutrition data.
