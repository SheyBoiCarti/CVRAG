# Streamlit Query Progress and Failure Handling Design

## Problem

The deployed app displays `Searching CV...` while both vector retrieval and NVIDIA answer generation run. A locally reproduced query, “Why should I hire this person?”, completed successfully but took roughly 33 seconds end to end. Because the same spinner remains visible for the entire operation, slow or queued generation looks like stalled retrieval.

The UI also catches every query exception and replaces it with a generic answer without logging the failure type. This prevents users from distinguishing a temporary provider problem from an unsupported CV question and makes deployment diagnosis unnecessarily difficult.

## Scope

This change will improve progress reporting and expected provider-failure handling for one submitted question. It will not introduce background workers, asynchronous job infrastructure, provider switching, or changes to CV retrieval relevance.

## Design

### Query stages

The engine will expose query preparation separately from answer streaming through a small public interface. Preparation will validate the question, optionally rewrite conversational follow-ups, retrieve and deduplicate CV chunks, and construct the evidence context. Generation will accept that prepared query and return the token iterator.

The existing synchronous and streaming query methods will reuse those operations so retrieval and answer construction continue to have one implementation.

### Streamlit presentation

For each question, the app will:

1. Display `Searching CV...` only while preparing the query and retrieving evidence.
2. Remove the search spinner once retrieval finishes.
3. If no evidence was found, immediately display the existing no-results answer.
4. Otherwise display `Generating answer...` while awaiting and streaming NVIDIA output.
5. Render source snippets using the retrieved evidence as before.

This makes a slow generation request visibly distinct from a retrieval delay without changing the conversational experience.

### Errors and observability

Expected upstream failures will produce a concise user-facing message explaining that the AI provider is temporarily unavailable and suggesting a retry. Unexpected failures will retain a generic safe message. Both paths will log the exception type and stage on the server without logging prompts, CV contents, or secrets.

The configured request timeout will continue to bound chat requests. The embedding wrapper will explicitly set the underlying NVIDIA client's timeout because the installed embedding class currently ignores a constructor-level `timeout` keyword and retains its 60-second default. Existing bounded retries remain in place.

### Compatibility

Current citations, conversation history, follow-up rewriting, deduplication, and synchronous `query()` behavior will remain unchanged. Existing callers of `query_stream()` will continue to work.

## Tests

Tests will be written before production changes and will verify:

- preparation and generation can be invoked as separate stages;
- empty retrieval returns the established no-results response without calling the LLM;
- streaming retains the current no-duplicate retry behavior;
- the configured embedding timeout reaches both NVIDIA transport clients;
- expected provider failures map to the intended safe UI-facing message or result.

The full offline test suite, syntax compilation, and dependency check will be run after implementation. The live diagnostic will be run once against the exact question to confirm retrieval and generation both complete, subject to provider availability.

## Success criteria

- The deployed UI never labels active answer generation as CV search.
- A provider timeout or transient endpoint failure terminates with a useful retry message rather than appearing indefinite.
- No exception details, API credentials, prompts, or CV content are exposed to visitors.
- Existing retrieval, conversation, streaming, and citation tests remain green.
