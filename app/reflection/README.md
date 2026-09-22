# app/reflection

The reflection companion is the `reflection_engine` package, pinned
in requirements.txt to a tagged release of
https://github.com/sugarlabs/reflection-engine (PyPI publication is
still an open question). For working on the engine itself, an
editable sibling checkout also works
(`uv pip install -e ../reflection-engine --python .venv/bin/python`).
This service imports it directly -- `Work`/`next_turn` plus the trace
codec (`as_record`/`from_record`) from `reflection_engine`.

The engine ships without its own provider adapters in play here: this
service already has model providers (`app/providers/`), and carrying
two provider stacks serves nobody. `bridge.py` is the whole adaptation
-- it wraps a sugar-ai `BaseProvider` so it satisfies the engine's
`complete(*, system, user, schema) -> dict` seam, asking for the
schema in the prompt and decoding the reply, since our providers have
no JSON mode of their own.

`schemas.py` is the one thing here that isn't the engine's: the HTTP
envelope (auth, quota, field ceilings) is transport, and transport
belongs to the service, not the engine. The conversation inside that
envelope travels as the engine's own trace records, untouched.

`app/routes/reflect.py` is the only place any of this gets called: it
builds the engine's `Work` (including Sugar's own
activity-bundle-ID-to-category table, which is host-specific and
doesn't travel with the engine), decodes the request's history records,
runs `next_turn`, and returns the result as one record.
