# SQLite baseline configuration

Benchmark result metadata identifies this file as the SQLite baseline's
`config_source`. The [SQLite reference notes](../../sqlite-baseline.md) document
the implementation, configuration and measurement boundaries.

The reference combines SQLite FTS5 with sqlite-vec scalar distance functions for
exact vector scoring. Text candidates are limited globally before Rust applies
metadata filters. Vector eligibility is determined before distance scoring.
Its candidate and fusion strategy is recorded separately from CofferDB's strategy.

The guide includes dependency configuration, failure behavior and timing limits.
Engine-local score values and process-startup boundaries must be interpreted with
those details when comparing results.
