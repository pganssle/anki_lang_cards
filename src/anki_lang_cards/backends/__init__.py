"""Model backends and the shared machinery for querying them.

A *model spec* such as ``anthropic/claude-sonnet-5`` is resolved to a
backend: dedicated backends claim specs by provider prefix, and anything
unclaimed falls back to the ``llm`` library, which routes to whatever model
plugins are installed. Backends declare optional :class:`base.Feature` s
(batching, prompt caching); :mod:`.runner` uses them, and :mod:`.selection`
decides which to turn on.
"""
