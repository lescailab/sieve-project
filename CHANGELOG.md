# Changelog

All notable changes to SIEVE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - "Stars End" - 2026-07-25

### Added
- Rank-quantile normalisation and FDR correction for gene–gene interaction scores.
- A `delta_rank` ranking metric, now the recommended metric for variant and gene ranking.
- A selectable classifier head (`flatten` or `attention_pool`), configurable and backward compatible.
- Conda packaging support for osx-arm64.
- Experimental documentation for a forthcoming carbon-footprint reporting feature (not yet part of the release).

### Changed
- Unified the project version across all packaging files to a single source of truth.
- Consolidated the documentation into a single source and improved its accuracy and build tooling.

## [1.2.0] - "Prime Radiant" - 2026-04-29

See the [GitHub release](https://github.com/lescailab/sieve-project/releases/tag/v1.2.0).

## [0.1.0] - "running dragon" - 2026-01-21

Initial pre-release. See the
[GitHub release](https://github.com/lescailab/sieve-project/releases/tag/0.1.0).

[1.3.0]: https://github.com/lescailab/sieve-project/releases/tag/v1.3.0
[1.2.0]: https://github.com/lescailab/sieve-project/releases/tag/v1.2.0
[0.1.0]: https://github.com/lescailab/sieve-project/releases/tag/0.1.0
